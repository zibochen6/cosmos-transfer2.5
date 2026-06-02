#!/usr/bin/env python3
"""
Batch Cosmos Transfer 2.5 augmentation for LeRobot 30s episode dataset.

Usage:
    cd /home/seeed/workspace/cosmos-transfer2.5
    source .venv/bin/activate

    # Full batch (all episodes, all cameras, all controls)
    python scripts/lerobot_side_augment.py --mode multicontrol

    # Flash attention mode
    COSMOS_USE_FLASH_ATTN=1 python scripts/lerobot_side_augment.py --mode multicontrol

    # Single-camera, resume, limit to first N episodes
    python scripts/lerobot_side_augment.py --mode multicontrol --cameras side --resume --limit 2

    # Edge-only mode
    python scripts/lerobot_side_augment.py --mode edge --cameras front side

Augmented videos are saved to:
    datasets/test_tube_30s/{camera}/episode-XXX_{group}.mp4
Control intermediates are saved to:
    datasets/test_tube_30s/controls/{episode}/{camera}/

The manifest is updated in-place with augmented_path columns.
"""

import argparse
import csv
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "assets/lerobot_example/prompts"
MANIFEST_DEFAULT = ROOT / "datasets/test_tube_30s/manifest.csv"

CAMERAS = ["observation.images.front", "observation.images.side"]
CAMERA_SHORT = {"front": "observation.images.front", "side": "observation.images.side"}
GROUPS = ["group_a", "group_b", "group_c", "group_d"]

sys.path.insert(0, str(ROOT))


def _check_video_codec(path: str) -> str:
    """Return codec name of a video file via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _transcode_to_h264(input_path: str, output_path: str):
    """Transcode any video to H264 for imageio compatibility."""
    print(f"    [TRANSCODE] {input_path} -> {output_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"    [TRANSCODE] failed: {r.stderr[-300:]}")
        sys.exit(1)
    print(f"    [TRANSCODE] done")


def _maybe_transcode_input(input_path: str) -> str:
    """If input is not H264/H265/VP9, transcode to H264. Returns path to use."""
    codec = _check_video_codec(input_path)
    if codec in ("h264", "hevc", "vp9"):
        return input_path
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _transcode_to_h264(input_path, tmp.name)
    return tmp.name  # caller must clean up


def _maybe_transcode_output(output_path: str) -> str | None:
    """If output codec is mp4v (OpenCV default), transcode to H264. Returns new path or None."""
    codec = _check_video_codec(output_path)
    if codec == "h264":
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _transcode_to_h264(output_path, tmp.name)
    # Replace the bad file
    shutil.move(tmp.name, output_path)
    return output_path


def resolve_controls(controls_dir: Path) -> dict:
    """Return existing control videos found in the controls directory."""
    existing = {}
    for f in controls_dir.glob("*.mp4"):
        existing[f.stem] = str(f)
    return existing


def compute_controls(
    input_video: str,
    controls_dir: Path,
    controls: list[str],
    seg_prompt: str = "robot arm gripper test tube rack laboratory",
    depth_encoder: str = "vits",
) -> dict:
    """Precompute control videos. Input should already be H264 (handled by process_episode)."""
    controls_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for ctype in controls:
        existing = resolve_controls(controls_dir)
        if ctype in existing:
            paths[ctype] = existing[ctype]
            print(f"    [{ctype.upper()}] skipped (already exists: {paths[ctype]})")
            continue

        output_path = str(controls_dir / f"{ctype}.mp4")

        if ctype == "edge":
            _run_edge(input_video, output_path)
        elif ctype == "vis":
            _run_vis(input_video, output_path)
        elif ctype == "depth":
            _run_depth(input_video, output_path, encoder=depth_encoder)
        elif ctype == "seg":
            _run_seg(input_video, output_path, prompt=seg_prompt)
        else:
            raise ValueError(f"Unknown control type: {ctype}")

        _maybe_transcode_output(output_path)
        paths[ctype] = output_path

    return paths


def _run_edge(input_path: str, output_path: str, t_lower: int = 100, t_upper: int = 200):
    """Compute Canny edge map video: ffmpeg pipe decode + OpenCV process + ffmpeg encode."""
    import cv2

    print(f"    [EDGE] {input_path} -> {output_path}")

    tmp_dir = Path(f"/tmp/edge_frames_{os.getpid()}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Decode AV1 (or any format) to PNG frames using ffmpeg
        decode_cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "none",
            "-i", input_path,
            "-f", "image2",
            str(tmp_dir / "frame_%06d.png"),
        ]
        r = subprocess.run(decode_cmd, capture_output=True)
        if r.returncode != 0:
            print(f"    [EDGE] decode failed: {r.stderr[-200:]}")
            sys.exit(1)

        frames = sorted(tmp_dir.glob("frame_*.png"))
        if not frames:
            print(f"    [EDGE] no frames decoded")
            sys.exit(1)

        # Read first frame to get fps/dims
        first = cv2.imread(str(frames[0]))
        height, width = first.shape[:2]
        fps = 30.0
        total = len(frames)
        print(f"    [EDGE] decoded {total} frames ({width}x{height})")

        # Process each frame: Canny edge
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            edges = cv2.Canny(blurred, t_lower, t_upper)
            cv2.imwrite(str(frame_path), edges)
            if (i + 1) % 200 == 0:
                print(f"      frame {i+1}/{total} ({100*(i+1)//total}%)")

        # Encode to libx264
        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]
        r = subprocess.run(encode_cmd, capture_output=True)
        if r.returncode != 0:
            print(f"    [EDGE] encode failed: {r.stderr[-300:]}")
            sys.exit(1)

        # Convert from single-channel yuvj420p to proper yuv420p (gray pixel
        # format is not widely supported; remux to normalize the container)
        tmp_fixed = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_fixed.close()
        fix_cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-c:v", "copy",
            "-pix_fmt", "yuv420p",
            tmp_fixed.name,
        ]
        r2 = subprocess.run(fix_cmd, capture_output=True)
        if r2.returncode == 0:
            shutil.move(tmp_fixed.name, output_path)
        else:
            Path(tmp_fixed.name).unlink(missing_ok=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    [EDGE] done: {output_path}")


def _run_vis(input_path: str, output_path: str):
    """Compute bilateral blur / vis control video: ffmpeg pipe decode + OpenCV process + ffmpeg encode."""
    import cv2

    print(f"    [VIS] {input_path} -> {output_path}")

    tmp_dir = Path(f"/tmp/vis_frames_{os.getpid()}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Decode input video to PNG frames
        decode_cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "none",
            "-i", input_path,
            "-f", "image2",
            str(tmp_dir / "frame_%06d.png"),
        ]
        r = subprocess.run(decode_cmd, capture_output=True)
        if r.returncode != 0:
            print(f"    [VIS] decode failed: {r.stderr[-200:]}")
            sys.exit(1)

        frames = sorted(tmp_dir.glob("frame_*.png"))
        if not frames:
            print(f"    [VIS] no frames decoded")
            sys.exit(1)

        first = cv2.imread(str(frames[0]))
        height, width = first.shape[:2]
        fps = 30.0
        d = 5
        sigma_color = 25.0
        sigma_space = 25.0
        downscale = 4

        print(f"    [VIS] decoded {len(frames)} frames ({width}x{height})")

        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            small = cv2.resize(frame, (width // downscale, height // downscale), interpolation=cv2.INTER_AREA)
            blurred = cv2.bilateralFilter(small, d, sigma_color, sigma_space)
            upscaled = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(frame_path), upscaled)

        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]
        r = subprocess.run(encode_cmd, capture_output=True)
        if r.returncode != 0:
            print(f"    [VIS] encode failed: {r.stderr[-300:]}")
            sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    [VIS] done: {output_path}")


def _apply_sdpa_patches():
    """Apply Thor-compatible SDPA patches to video_depth_anything.

    Thor (sm_90) lacks xformers flash attention compiled kernels.
    video_depth_anything uses xformers in two places:
      1. dinov2_layers/attention.py: MemEffAttention.forward
      2. motion_module/motion_module.py: TemporalAttention.forward
    Both are patched to use PyTorch F.scaled_dot_product_attention.
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat

    # Patch 1: dinov2_layers/attention.py
    import video_depth_anything.dinov2_layers.attention as attn_mod

    def patched_dinov2_fwd(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)

    attn_mod.MemEffAttention.forward = patched_dinov2_fwd

    # Patch 2: motion_module/motion_module.py - TemporalAttention
    import video_depth_anything.motion_module.motion_module as motion_mod

    _orig_ta_init = motion_mod.TemporalAttention.__init__

    def patched_ta_init(self, *args, **kwargs):
        _orig_ta_init(self, *args, **kwargs)
        self._use_memory_efficient_attention_xformers = False

    motion_mod.TemporalAttention.__init__ = patched_ta_init

    def patched_ta_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None,
                           video_length=None, cached_hidden_states=None):
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]
        d_in = 0
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            input_hidden_states = hidden_states
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            input_hidden_states = hidden_states
            d_in = cached_hidden_states.shape[1]
            hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = repeat(
            encoder_hidden_states, "b n c -> (b d) n c", d=d
        ) if encoder_hidden_states is not None else encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states[:, d_in:, ...])

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            from video_depth_anything.motion_module.attention import apply_rotary_emb
            query, key = apply_rotary_emb(query, key, freqs_cis)

        B_d, F_in, C_total = query.shape
        H = self.heads
        D = C_total // H

        q = query.reshape(B_d, F_in, H, D).transpose(1, 2)
        k = key.reshape(B_d, F_in, H, D).transpose(1, 2)
        v = value.reshape(B_d, F_in, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B_d, F_in, C_total)

        out = self.to_out[0](out)
        out = self.to_out[1](out)

        f_out = F_in
        out = rearrange(out, "(b d) f c -> (b f) d c", b=input_hidden_states.shape[0] // d, d=d, f=f_out)
        if cached_hidden_states is None:
            residual = rearrange(input_hidden_states, "(b d) f c -> (b f) d c",
                                b=input_hidden_states.shape[0] // d, d=d, f=video_length)
        else:
            raise NotImplementedError("cached_hidden_states not needed for batch inference")
        return out + residual, out

    motion_mod.TemporalAttention.forward = patched_ta_forward
    print("    [PATCH] video_depth_anything: xformers -> PyTorch SDPA (Thor compatible)")


def _run_depth(input_path: str, output_path: str, encoder: str = "vits"):
    print(f"    [DEPTH] {input_path} -> {output_path}")

    # Write patch script to a temp file (avoids f-string escaping issues in inline -c)
    patch_script = f"""
import sys
sys.path.insert(0, '{ROOT}')
from scripts.lerobot_side_augment import _apply_sdpa_patches
_apply_sdpa_patches()

import importlib.util
spec = importlib.util.spec_from_file_location(
    'depth_pipeline',
    '{ROOT}/cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py'
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import argparse, sys
sys.argv = [
    'depth_pipeline',
    '--input_video', '{input_path}',
    '--output_video', '{output_path}',
    '--encoder', '{encoder}',
]
m.main()
"""
    script_path = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    script_path.write(patch_script)
    script_path.close()

    try:
        cmd = [sys.executable, script_path.name]
        result = subprocess.run(cmd, cwd=str(ROOT), env=_clean_env())
        if result.returncode != 0:
            print(f"    [DEPTH] FAILED")
            print(f"    stderr: {result.stderr[-500:]}")
            sys.exit(result.returncode)
    finally:
        Path(script_path.name).unlink(missing_ok=True)

    print(f"    [DEPTH] done: {output_path}")


def _run_seg(input_path: str, output_path: str, prompt: str = "robot arm gripper test tube rack laboratory"):
    print(f"    [SEG] {input_path} -> {output_path} (prompt: '{prompt}')")
    cmd = [
        sys.executable,
        str(ROOT / "cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py"),
        "--input_video", input_path,
        "--output_video", output_path,
        "--mode", "prompt",
        "--prompt", prompt,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), env=_clean_env())
    if result.returncode != 0:
        print(f"    [SEG] FAILED")
        sys.exit(result.returncode)
    print(f"    [SEG] done: {output_path}")


NEGATIVE_PROMPT = (
    "The video captures a game playing, with bad crappy graphics and cartoonish frames. "
    "It represents a recording of old outdated games. The lighting looks very fake. "
    "The textures are very raw and basic. The geometries are very primitive. "
    "The images are very pixelated and of poor CG quality. "
    "There are many subtitles in the footage. Overall, the video is unrealistic at all."
)


def build_spec_json(
    mode: str,
    input_video: str,
    prompt_path: str,
    output_video: str,
    control_paths: dict | None = None,
    edge_weight: float = 1.0,
    depth_weight: float = 1.0,
    seg_weight: float = 0.5,
    vis_weight: float = 0.5,
    guidance: int = 3,
) -> dict:
    """Build a temporary spec dict for inference."""
    spec = {
        "name": "batch_augment",
        "prompt_path": prompt_path,
        "video_path": input_video,
        "guidance": guidance,
        "negative_prompt": NEGATIVE_PROMPT,
    }

    if mode == "edge":
        spec["edge"] = {
            "control_path": control_paths.get("edge", ""),
            "control_weight": edge_weight,
        }

    elif mode == "multicontrol":
        spec["depth"] = {
            "control_path": control_paths.get("depth", ""),
            "control_weight": depth_weight,
        }
        spec["edge"] = {
            "control_path": control_paths.get("edge", ""),
            "control_weight": edge_weight,
        }
        spec["seg"] = {
            "control_path": control_paths.get("seg", ""),
            "control_weight": seg_weight,
        }
        spec["vis"] = {
            "control_path": control_paths.get("vis", ""),
            "control_weight": vis_weight,
        }

    return spec


def write_spec_to_temp(spec: dict) -> Path:
    """Write spec dict to a temporary JSON file. Returns the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
    json.dump(spec, tmp, indent=4)
    tmp.flush()
    return Path(tmp.name)


def _clean_env():
    """Return a copy of os.environ with proxy vars removed."""
    import re as _proxy_re
    env = dict(os.environ)
    _proxy_keys = _proxy_re.compile(
        r"^(https?|http|socks|all|no)_?(proxy)?$", _proxy_re.IGNORECASE
    )
    for k in list(env):
        if _proxy_keys.match(k):
            del env[k]
    return env


def run_inference(spec_path: Path, output_dir: Path) -> int:
    """Run examples/inference.py. Returns the subprocess return code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "examples/inference.py"),
        "-i", str(spec_path),
        "-o", str(output_dir),
    ]
    print(f"    [INFER] {spec_path.name} -> {output_dir}")
    result = subprocess.run(cmd, cwd=str(ROOT), env=_clean_env())
    if result.returncode != 0:
        print(f"    [INFER] FAILED with code {result.returncode}")
    return result.returncode


def process_episode(
    episode_name: str,
    episode_rows: list[dict],
    cameras: list[str],
    mode: str,
    controls: list[str],
    prompt_group: str,
    controls_base: Path,
    output_base: Path,
    resume: bool,
) -> dict[str, str]:
    """Process a single episode for specified cameras. Returns {camera: output_path}."""
    results = {}
    episode_controls_dir = controls_base / episode_name

    for row in episode_rows:
        camera = row["camera"]
        if camera not in cameras:
            continue

        input_video = str(ROOT / "datasets" / "test_tube_30s" / camera / f"{episode_name}.mp4")

        if not Path(input_video).exists():
            print(f"  [WARN] Input not found: {input_video}, skipping")
            continue

        cam_controls_dir = episode_controls_dir / camera
        cam_output_dir = output_base / camera

        output_name = f"{episode_name}_{prompt_group}.mp4"
        output_path = cam_output_dir / output_name

        if resume and output_path.exists():
            print(f"  [{episode_name}/{camera}] already exists, skipping (--resume)")
            results[camera] = str(output_path)
            continue

        cam_controls_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = PROMPTS_DIR / prompt_group / f"{camera.split('.')[-1]}.txt"
        if not prompt_file.exists():
            prompt_file = PROMPTS_DIR / prompt_group / "side.txt"

        # Transcode AV1 input to H264 once for both controls and inference
        actual_input = _maybe_transcode_input(input_video)
        is_transcoded_input = actual_input != input_video
        try:
            control_paths = compute_controls(
                input_video=actual_input,
                controls_dir=cam_controls_dir,
                controls=controls,
            )

            spec = build_spec_json(
                mode=mode,
                input_video=actual_input,
                prompt_path=str(prompt_file),
                output_video=str(output_path),
                control_paths=control_paths,
            )

            spec_path = write_spec_to_temp(spec)
            try:
                code = run_inference(spec_path, cam_output_dir)
                if code == 0:
                    results[camera] = str(output_path)
                else:
                    print(f"  [{episode_name}/{camera}] inference failed, skipping manifest update")
            finally:
                spec_path.unlink(missing_ok=True)
        finally:
            if is_transcoded_input:
                Path(actual_input).unlink(missing_ok=True)

    return results


def update_manifest(manifest_path: Path, augmented_map: dict):
    """Update augmented_path in manifest CSV for processed rows."""
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            key = (row["episode"], row["camera"])
            if key in augmented_map:
                row["augmented_path"] = augmented_map[key]
            rows.append(row)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [MANIFEST] updated: {len(augmented_map)} paths written")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    parser.add_argument(
        "--mode",
        choices=["edge", "multicontrol"],
        default="multicontrol",
        help="edge = edge-only control; multicontrol = depth+edge+seg+vis",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="front,side",
        help="Comma-separated cameras to process (default: front,side)",
    )
    parser.add_argument(
        "--controls",
        type=str,
        default="edge,depth,vis,seg",
        help="Comma-separated control types to precompute (default: edge,depth,vis,seg)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip episodes that already have an augmented output")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to the first N episodes (0 = no limit)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for prompt group selection (default: time-based)",
    )
    parser.add_argument(
        "--depth_encoder",
        type=str,
        choices=["vits", "vitl"],
        default="vits",
        help="Video-Depth-Anything encoder (default: vits)",
    )
    parser.add_argument(
        "--seg_prompt",
        type=str,
        default="robot arm gripper test tube rack laboratory",
        help="Text prompt for SAM2 segmentation",
    )
    args = parser.parse_args()

    cameras = [CAMERA_SHORT.get(c.strip(), c.strip()) for c in args.cameras.split(",")]
    controls = [c.strip() for c in args.controls.split(",")]
    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    controls_base = manifest_path.parent / "controls"
    output_base = manifest_path.parent

    rows_by_episode = defaultdict(list)
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_episode[row["episode"]].append(row)

    episode_names = sorted(rows_by_episode.keys())
    if args.limit > 0:
        episode_names = episode_names[: args.limit]

    print(f"\n{'='*60}")
    print(f"Batch Augmentation — {len(episode_names)} episodes x {cameras}")
    print(f"  Mode:    {args.mode}")
    print(f"  Cameras: {cameras}")
    print(f"  Controls: {controls}")
    print(f"  Resume:  {'yes' if args.resume else 'no'}")
    print(f"  Random seed: {seed}")
    print(f"  Flash attn:  {'ON' if os.environ.get('COSMOS_USE_FLASH_ATTN') == '1' else 'OFF'}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}\n")

    total_augmented = {}

    for ep_idx, episode_name in enumerate(episode_names):
        episode_rows = rows_by_episode[episode_name]
        prompt_group = random.choice(GROUPS)

        print(f"[{ep_idx+1}/{len(episode_names)}] Episode: {episode_name} | prompt: {prompt_group}")
        t0 = time.time()

        augmented = process_episode(
            episode_name=episode_name,
            episode_rows=episode_rows,
            cameras=cameras,
            mode=args.mode,
            controls=controls,
            prompt_group=prompt_group,
            controls_base=controls_base,
            output_base=output_base,
            resume=args.resume,
        )

        elapsed = time.time() - t0
        print(f"[{episode_name}] done in {elapsed:.0f}s | outputs: {list(augmented.keys())}")

        total_augmented.update(augmented)

        if augmented:
            update_manifest(manifest_path, augmented)

    print(f"\n{'='*60}")
    print(f"All done! Processed {len(episode_names)} episodes")
    print(f"Total outputs: {len(total_augmented)}")
    print(f"Controls saved to: {controls_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
