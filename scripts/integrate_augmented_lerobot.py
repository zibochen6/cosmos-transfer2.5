#!/usr/bin/env python3
"""
Integrate augmented video episodes back into a LeRobot training-ready dataset.

This script:
  1. Reads manifest.csv to find episodes with generated augmented videos.
  2. Extracts the original robot state/action data (parquet) for each episode.
  3. Generates a new LeRobot dataset directory with:
       - meta/{info,stats,episodes,modality}.json
       - data/chunk-XXX/file-XXX.parquet  (aligned to augmented episodes)
       - videos/{camera}/chunk-XXX/file-XXX.mp4
         - observation.images.front: hard-links to original video
         - observation.images.side: copies of augmented videos
  4. Re-computes statistics across all integrated episodes.

Usage:
    python scripts/integrate_augmented_lerobot.py

Configure paths at the top of the script:
    ORIGINAL_LEROBOT_ROOT   - LeRobot dataset with .parquet data files
    AUGMENTED_DATASET_ROOT  - Output of split_dataset_30s.py + lerobot_side_augment.py
    OUTPUT_ROOT             - Final training-ready dataset
"""

import csv
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# =========================================================================
# CONFIGURE THESE PATHS
# =========================================================================
ORIGINAL_LEROBOT_ROOT = ROOT / "lerobot/seeed_rebot_b601_dm/organize_test_tube"
AUGMENTED_DATASET_ROOT = ROOT / "datasets/test_tube_30s"
OUTPUT_ROOT = ROOT / "outputs/seeed_rebot_b601_dm_augmented"
# =========================================================================

MANIFEST_PATH = AUGMENTED_DATASET_ROOT / "manifest.csv"
PARQUET_CHUNK_SIZE = 1000
FPS = 30.0
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_manifest() -> dict[str, list[dict]]:
    """Return {episode_name: [row, ...]} for all rows with non-empty augmented_path."""
    rows_by_episode = defaultdict(list)
    augmented_episodes = set()
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_episode[row["episode"]].append(row)
            if row.get("augmented_path", "").strip():
                augmented_episodes.add(row["episode"])

    print(f"Manifest: {len(rows_by_episode)} episodes, {len(augmented_episodes)} have augmented videos")
    return rows_by_episode, augmented_episodes


def episode_to_parquet_info(episode_name: str) -> tuple[int, int, float, float]:
    """Parse file_index, start_sec, end_sec from episode name.

    episode format: episode-{idx}_file-{file_index}_{start}-{end}s[_remainder]
    Returns (file_index, chunk_index, start_sec, end_sec)
    chunk_index is always 0 for this project.
    """
    m = re.search(r"_file-(\d+)_(\d+\.\d+)-(\d+\.\d+)s", episode_name)
    if not m:
        raise ValueError(f"Cannot parse episode name: {episode_name}")
    file_index = int(m.group(1))
    start_sec = float(m.group(2))
    end_sec = float(m.group(3))
    return file_index, 0, start_sec, end_sec


def find_original_parquet(file_index: int, chunk_index: int = 0) -> Path:
    """Return path to the original parquet file for the given file/chunk index."""
    pq = ORIGINAL_LEROBOT_ROOT / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"Original parquet not found: {pq}")
    return pq


def find_original_video(camera: str, file_index: int, chunk_index: int = 0) -> Path:
    """Return path to the original video for the given camera/file/chunk."""
    vid = ORIGINAL_LEROBOT_ROOT / "videos" / camera / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    if not vid.exists():
        raise FileNotFoundError(f"Original video not found: {vid}")
    return vid


def extract_episode_parquet(
    parquet_path: Path,
    start_sec: float,
    end_sec: float,
    episode_global_index: int,
) -> pd.DataFrame:
    """Extract rows from parquet whose timestamp falls within [start_sec, end_sec].

    Returns a DataFrame with re-indexed columns:
        index, episode_index, frame_index, timestamp, task_index,
        action, observation.state, observation.images.front, observation.images.side
    The video reference columns point to relative paths from the OUTPUT_ROOT.
    """
    df = pd.read_parquet(parquet_path)

    COL_TIMESTAMP = "timestamp"
    COL_FRAME = "frame_index"
    COL_EPISODE = "episode_index"
    COL_TASK = "task_index"
    COL_ACTION = "action"
    COL_STATE = "observation.state"

    ts_col = df[COL_TIMESTAMP]
    mask = (ts_col >= start_sec) & (ts_col <= end_sec + 1e-3)
    episode_df = df[mask].copy()

    if episode_df.empty:
        raise ValueError(f"No parquet rows found for [{start_sec}, {end_sec}] in {parquet_path}")

    n = len(episode_df)
    episode_df["episode_index"] = episode_global_index
    episode_df["frame_index"] = np.arange(n, dtype=np.int64)
    episode_df["index"] = np.arange(n, dtype=np.int64)

    return episode_df


def build_output_parquet_row(
    original_row: pd.Series,
    camera: str,
    file_index: int,
    chunk_index: int,
    relative_video_path: str,
) -> dict:
    """Build a dict for a single parquet row with updated video reference."""
    return {
        "index": int(original_row["index"]),
        "episode_index": int(original_row["episode_index"]),
        "frame_index": int(original_row["frame_index"]),
        "timestamp": float(original_row["timestamp"]),
        "task_index": int(original_row["task_index"]) if "task_index" in original_row else 0,
        "action": np.array(original_row["action"], dtype=np.float32),
        "observation.state": np.array(original_row["observation.state"], dtype=np.float32),
        "observation.images.front": f"../videos/observation.images.front/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "observation.images.side": relative_video_path,
    }


def compute_stats(frames: list[dict]) -> dict:
    """Compute min/max/mean/std/q01-q99 for action and observation.state."""
    actions = np.array([f["action"] for f in frames], dtype=np.float32)
    states = np.array([f["observation.state"] for f in frames], dtype=np.float32)

    def _stats(arr: np.ndarray) -> dict:
        return {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    return {
        "action": _stats(actions),
        "observation.state": _stats(states),
    }


def write_parquet_and_video(
    episode_name: str,
    rows: list[dict],
    file_index: int,
    chunk_index: int,
    stats_collector: list,
    output_data_dir: Path,
    output_video_dir: Path,
    original_lerobot_root: Path,
):
    """Write a single parquet file + copy/hardlink videos for one episode.

    rows: list of manifest rows for this episode (one per camera).
    """
    out_data_chunk = output_data_dir / f"chunk-{chunk_index:03d}"
    out_data_chunk.mkdir(parents=True, exist_ok=True)
    out_video_front = output_video_dir / "observation.images.front" / f"chunk-{chunk_index:03d}"
    out_video_side = output_video_dir / "observation.images.side" / f"chunk-{chunk_index:03d}"
    out_video_front.mkdir(parents=True, exist_ok=True)
    out_video_side.mkdir(parents=True, exist_ok=True)

    out_parquet_path = out_data_chunk / f"file-{file_index:03d}.parquet"

    front_video_path = None
    side_video_path = None
    episode_frames = []

    for row in rows:
        camera = row["camera"]
        aug_path = row.get("augmented_path", "").strip()
        _, _, start_sec, end_sec = episode_to_parquet_info(episode_name)

        pq_path = find_original_parquet(file_index, chunk_index)
        ep_df = extract_episode_parquet(pq_path, start_sec, end_sec, episode_global_index=0)

        if camera == "observation.images.front":
            orig_vid = find_original_video(camera, file_index, chunk_index)
            front_video_path = orig_vid
            rel_front = f"../videos/observation.images.front/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

            for _, orig_row in ep_df.iterrows():
                frame = {
                    "action": orig_row["action"],
                    "observation.state": orig_row["observation.state"],
                }
                episode_frames.append(frame)

        elif camera == "observation.images.side":
            if not aug_path:
                print(f"  [WARN] {episode_name} has no augmented_path for side, skipping")
                continue
            aug_src = Path(aug_path)
            side_rel = f"../videos/observation.images.side/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            side_video_path = out_video_side / f"file-{file_index:03d}.mp4"

            if not side_video_path.exists():
                shutil.copy2(aug_src, side_video_path)
                print(f"    [VIDEO] copied augmented side -> {side_video_path}")

    # Hard-link original front video if it exists
    if front_video_path and front_video_path.exists():
        out_front = out_video_front / f"file-{file_index:03d}.mp4"
        if not out_front.exists():
            os.link(front_video_path, out_front)
            print(f"    [VIDEO] hard-linked original front -> {out_front}")

    if not episode_frames:
        print(f"  [WARN] No frames collected for {episode_name}, skipping parquet")
        return

    all_actions = np.array([f["action"] for f in episode_frames], dtype=np.float32)
    all_states = np.array([f["observation.state"] for f in episode_frames], dtype=np.float32)
    n = len(episode_frames)

    parquet_records = []
    for i, orig_row in ep_df.iterrows():
        front_rel = f"../videos/observation.images.front/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        side_rel = f"../videos/observation.images.side/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        parquet_records.append({
            "index": i if isinstance(i, (int, np.integer)) else i,
            "episode_index": 0,
            "frame_index": len(parquet_records),
            "timestamp": float(orig_row["timestamp"]),
            "task_index": int(orig_row["task_index"]) if "task_index" in orig_row else 0,
            "action": np.array(orig_row["action"], dtype=np.float32),
            "observation.state": np.array(orig_row["observation.state"], dtype=np.float32),
            "observation.images.front": front_rel,
            "observation.images.side": side_rel,
        })

    pq_df = pd.DataFrame(parquet_records)
    pq_df.to_parquet(out_parquet_path, index=False)
    print(f"    [PARQUET] wrote {n} rows -> {out_parquet_path}")

    stats_collector.append({
        "action": all_actions,
        "observation.state": all_states,
    })


def write_metadata(
    output_root: Path,
    episode_counts: list[int],
    all_stats: list,
    original_info: dict,
):
    """Write meta/info.json, meta/episodes.jsonl, meta/stats.json, meta/modality.json."""
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- info.json ---
    info = {
        "codebase_version": "v3.0",
        "robot_type": original_info.get("robot_type", "seeed_b601_dm_follower"),
        "total_episodes": len(episode_counts),
        "total_frames": sum(episode_counts),
        "total_tasks": 1,
        "chunks_size": PARQUET_CHUNK_SIZE,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": FPS,
        "splits": {"train": f"0:{len(episode_counts)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": original_info.get("features", _default_features()),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- episodes.jsonl ---
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_idx, count in enumerate(episode_counts):
            f.write(json.dumps({"episode_index": ep_idx, "length": count}) + "\n")

    # --- stats.json ---
    all_actions = np.concatenate([s["action"] for s in all_stats], axis=0)
    all_states = np.concatenate([s["observation.state"] for s in all_stats], axis=0)

    def _col_stats(arr: np.ndarray) -> dict:
        return {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
            "count": [len(arr)],
        }

    stats = {
        "action": _col_stats(all_actions),
        "observation.state": _col_stats(all_states),
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # --- modality.json ---
    modality = {
        "state": {
            name: {
                "start": i,
                "end": i + 1,
                "absolute": True,
                "dtype": "float32",
            }
            for i, name in enumerate(JOINT_NAMES)
        },
        "action": {
            name: {
                "start": i,
                "end": i + 1,
                "absolute": True,
                "dtype": "float32",
            }
            for i, name in enumerate(JOINT_NAMES)
        },
        "video": {
            "observation.images.front": {},
            "observation.images.side": {},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    print(f"  [META] info.json, episodes.jsonl, stats.json, modality.json written")


def _default_features() -> dict:
    """Return default LeRobot feature schema when original info.json is unavailable."""
    return {
        "action": {
            "dtype": "float32",
            "names": JOINT_NAMES,
            "shape": [7],
        },
        "observation.state": {
            "dtype": "float32",
            "names": JOINT_NAMES,
            "shape": [7],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def main():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found: {MANIFEST_PATH}")
        return

    original_info = {}
    info_path = ORIGINAL_LEROBOT_ROOT / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            original_info = json.load(f)
        print(f"Loaded original info.json: robot_type={original_info.get('robot_type')}")
    else:
        print(f"[WARN] Original info.json not found at {info_path}, using default schema")

    rows_by_episode, augmented_episodes = parse_manifest()

    if not augmented_episodes:
        print("ERROR: No episodes have augmented videos. Run lerobot_side_augment.py first.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_data_dir = OUTPUT_ROOT / "data"
    output_video_dir = OUTPUT_ROOT / "videos"

    stats_collector = []
    episode_counts = []
    episode_global_index = 0

    # Group episodes by (file_index, chunk_index) to batch into single parquet files
    by_file_chunk = defaultdict(list)

    for episode_name in sorted(augmented_episodes):
        rows = rows_by_episode.get(episode_name, [])
        if not rows:
            continue
        file_index, chunk_index, _, _ = episode_to_parquet_info(episode_name)
        by_file_chunk[(file_index, chunk_index)].append((episode_name, rows))

    # Process each (file_index, chunk_index) group as one parquet file
    for (file_index, chunk_index), episodes in sorted(by_file_chunk.items()):
        print(f"\nProcessing chunk-{chunk_index:03d}/file-{file_index:03d}: {len(episodes)} episodes")

        all_parquet_records = []
        episode_frame_counts = []

        for episode_name, rows in episodes:
            _, _, start_sec, end_sec = episode_to_parquet_info(episode_name)
            pq_path = find_original_parquet(file_index, chunk_index)
            ep_df = extract_episode_parquet(pq_path, start_sec, end_sec, episode_global_index=episode_global_index)

            front_rel = f"../videos/observation.images.front/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            side_rel = f"../videos/observation.images.side/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

            for idx, (_, orig_row) in enumerate(ep_df.iterrows()):
                all_parquet_records.append({
                    "index": idx,
                    "episode_index": episode_global_index,
                    "frame_index": idx,
                    "timestamp": float(orig_row["timestamp"]),
                    "task_index": int(orig_row["task_index"]) if "task_index" in orig_row else 0,
                    "action": np.array(orig_row["action"], dtype=np.float32),
                    "observation.state": np.array(orig_row["observation.state"], dtype=np.float32),
                    "observation.images.front": front_rel,
                    "observation.images.side": side_rel,
                })

            episode_frame_counts.append(len(ep_df))
            episode_global_index += 1
            print(f"  [{episode_name}] frames={len(ep_df)}")

            # Copy video files
            front_vid = find_original_video("observation.images.front", file_index, chunk_index)
            out_front_dir = output_video_dir / "observation.images.front" / f"chunk-{chunk_index:03d}"
            out_front_dir.mkdir(parents=True, exist_ok=True)
            out_front = out_front_dir / f"file-{file_index:03d}.mp4"
            if not out_front.exists():
                os.link(front_video_path := front_vid, out_front)
                print(f"    [VIDEO] hard-linked front -> {out_front}")

            for row in rows:
                camera = row["camera"]
                aug_path = row.get("augmented_path", "").strip()
                if camera == "observation.images.side" and aug_path:
                    out_side_dir = output_video_dir / "observation.images.side" / f"chunk-{chunk_index:03d}"
                    out_side_dir.mkdir(parents=True, exist_ok=True)
                    out_side = out_side_dir / f"file-{file_index:03d}.mp4"
                    if not out_side.exists():
                        shutil.copy2(Path(aug_path), out_side)
                        print(f"    [VIDEO] copied augmented side -> {out_side}")

        # Write parquet for this file chunk
        out_data_chunk = output_data_dir / f"chunk-{chunk_index:03d}"
        out_data_chunk.mkdir(parents=True, exist_ok=True)
        out_parquet = out_data_chunk / f"file-{file_index:03d}.parquet"
        pq_df = pd.DataFrame(all_parquet_records)
        pq_df.to_parquet(out_parquet, index=False)
        print(f"  [PARQUET] {len(pq_df)} total rows -> {out_parquet}")

        # Collect stats
        actions = np.array([r["action"] for r in all_parquet_records], dtype=np.float32)
        states = np.array([r["observation.state"] for r in all_parquet_records], dtype=np.float32)
        stats_collector.append({"action": actions, "observation.state": states})
        episode_counts.extend(episode_frame_counts)

    print(f"\n{'='*60}")
    print(f"Total episodes integrated: {episode_global_index}")
    print(f"Total frames: {sum(episode_counts)}")

    write_metadata(OUTPUT_ROOT, episode_counts, stats_collector, original_info)

    print(f"\nOutput dataset: {OUTPUT_ROOT}")
    print("Done!")


if __name__ == "__main__":
    main()
