#!/usr/bin/env python3
"""
Split lerobot video dataset into 30-second episodes with front/side camera alignment.
Outputs to datasets/test_tube_30s/ with a manifest.csv for later concatenation.
"""

import csv
import subprocess
from pathlib import Path

ROOT = Path("/home/seeed/workspace/cosmos-transfer2.5")
INPUT_DIR = ROOT / "lerobot/seeed_rebot_b601_dm/organize_test_tube/videos"
OUTPUT_DIR = ROOT / "datasets/test_tube_30s"
CHUNK_DURATION = 30.0  # seconds per episode

# Camera names match directory structure under INPUT_DIR
CAMERAS = ["observation.images.front", "observation.images.side"]
SOURCE_FILES = ["file-000.mp4", "file-001.mp4"]


def get_video_info(path: Path) -> dict:
    """Use ffprobe to get exact duration and frame count."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration,nb_frames,r_frame_rate",
            "-of", "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return {
        "duration": float(info["duration"]),
        "nb_frames": int(info["nb_frames"]),
        "fps": eval(info["r_frame_rate"].replace("/", ".")) if "r_frame_rate" in info else None,
    }


def split_video(input_path: Path, start: float, end: float, output_path: Path) -> int:
    """Split video using ffmpeg -c copy (no re-encoding). Returns actual frame count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(input_path),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    info = get_video_info(output_path)
    return info["nb_frames"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create output subdirs
    for cam in CAMERAS:
        (OUTPUT_DIR / cam).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    episode_idx = 0

    for source_file in SOURCE_FILES:
        # Load duration from the first camera (they should be synced)
        cam_paths = {cam: INPUT_DIR / cam / "chunk-000" / source_file for cam in CAMERAS}
        ref_cam = CAMERAS[0]
        ref_path = cam_paths[ref_cam]
        info = get_video_info(ref_path)
        total_duration = info["duration"]
        total_frames = info["nb_frames"]
        fps = info["fps"]

        print(f"\n{'='*60}")
        print(f"Source: {source_file}")
        print(f"  Duration: {total_duration:.2f}s, Frames: {total_frames}, FPS: {fps:.2f}")

        # Generate split segments
        seg_idx = 0
        seg_start = 0.0
        while seg_start < total_duration:
            seg_end = min(seg_start + CHUNK_DURATION, total_duration)
            is_remainder = (seg_end - seg_start) < CHUNK_DURATION
            duration = seg_end - seg_start

            episode_name = f"episode-{episode_idx:03d}_{source_file.replace('.mp4','')}_{seg_start:.0f}-{seg_end:.0f}s"
            episode_name += "_remainder" if is_remainder else ""

            print(f"  [{episode_idx:02d}] {seg_start:.1f}s - {seg_end:.1f}s ({duration:.1f}s)"
                  + (" [REMAINDER]" if is_remainder else ""))

            for cam in CAMERAS:
                input_path = cam_paths[cam]
                output_path = OUTPUT_DIR / cam / f"{episode_name}.mp4"
                frame_count = split_video(input_path, seg_start, seg_end, output_path)

                manifest_rows.append({
                    "episode": episode_name,
                    "camera": cam,
                    "source_file": source_file,
                    "start_sec": round(seg_start, 3),
                    "end_sec": round(seg_end, 3),
                    "frame_count": frame_count,
                    "is_remainder": "yes" if is_remainder else "no",
                    "augmented_path": "",
                    "augmented_fps": fps,
                })

            episode_idx += 1
            seg_start = seg_end

    # Write manifest.csv
    manifest_path = OUTPUT_DIR / "manifest.csv"
    fieldnames = ["episode", "camera", "source_file", "start_sec", "end_sec",
                  "frame_count", "is_remainder", "augmented_path", "augmented_fps"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{'='*60}")
    print(f"Done! Total episodes: {episode_idx}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Manifest:   {manifest_path}")
    print(f"Total manifest rows: {len(manifest_rows)}")

    # Quick sanity check
    front_count = sum(1 for r in manifest_rows if r["camera"] == CAMERAS[0])
    side_count = sum(1 for r in manifest_rows if r["camera"] == CAMERAS[1])
    print(f"Front segments: {front_count}, Side segments: {side_count}")

    # Verify front/side alignment
    mismatches = 0
    for front_row in [r for r in manifest_rows if r["camera"] == CAMERAS[0]]:
        matching_side = next(
            (r for r in manifest_rows
             if r["camera"] == CAMERAS[1]
             and r["episode"] == front_row["episode"]),
            None,
        )
        if matching_side:
            if front_row["start_sec"] != matching_side["start_sec"] or \
               front_row["end_sec"] != matching_side["end_sec"]:
                mismatches += 1
                print(f"  MISMATCH: {front_row['episode']}")
    if mismatches == 0:
        print("Front/Side alignment check: PASSED")
    else:
        print(f"Front/Side alignment check: FAILED ({mismatches} mismatches)")


if __name__ == "__main__":
    main()
