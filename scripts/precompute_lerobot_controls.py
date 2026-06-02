#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-compute all control videos for LeRobot side-camera input.
This script generates edge, depth, vis (blur), and seg (segmentation) control videos
from the input RGB video, saving them to local files for inspection before running
the full Cosmos Transfer 2.5 inference.

Each control type is computed independently so you can inspect results separately.

Usage:
    # Compute ALL controls (depth + edge + vis + seg)
    python scripts/precompute_lerobot_controls.py --all

    # Compute individual controls
    python scripts/precompute_lerobot_controls.py --edge
    python scripts/precompute_lerobot_controls.py --depth
    python scripts/precompute_lerobot_controls.py --vis
    python scripts/precompute_lerobot_controls.py --seg

    # Custom input
    python scripts/precompute_lerobot_controls.py --all \
        --input_video /path/to/video.mp4 \
        --output_dir /path/to/output_folder
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# 1. EDGE — Canny Edge Detection (pure OpenCV, no model download)
# ============================================================================


def compute_edge_video(input_path: str, output_path: str, t_lower: int = 100, t_upper: int = 200):
    """Compute Canny edge map video from RGB input.

    Args:
        input_path: Path to input RGB video.
        output_path: Path to save edge video (.mp4).
        t_lower: Canny lower threshold (lower = more edges, more noise).
        t_upper: Canny upper threshold (upper = fewer but stronger edges).
    """
    print(f"\n{'='*60}")
    print(f"[EDGE] Computing Canny edge video")
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    print(f"  Thresholds: lower={t_lower}, upper={t_upper}")
    print(f"  (Lower threshold detects more edges; higher = cleaner edges)")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, t_lower, t_upper)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        out.write(edges_3ch)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / frame_idx) * (total_frames - frame_idx)
            print(f"  Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    cap.release()
    out.release()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done! {frame_idx} frames in {elapsed:.1f}s | {size_mb:.1f} MB -> {output_path}")


# ============================================================================
# 2. VIS — Bilateral Blur (pure OpenCV, no model download)
# ============================================================================


def compute_vis_video(
    input_path: str,
    output_path: str,
    d: int = 30,
    sigma_color: float = 150.0,
    sigma_space: float = 100.0,
    downscale_factor: int = 4,
):
    """Compute bilateral blur / guided filter style vis control video.

    This replicates the AddControlInputBlur augmentor's inference preset:
    - Downscale by downscale_factor
    - Apply bilateral filter
    - Upscale back with bicubic interpolation

    Args:
        input_path: Path to input RGB video.
        output_path: Path to save vis video (.mp4).
        d: Diameter of pixel neighborhood for bilateral filter.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.
        downscale_factor: How much to downscale before blur (1 = no downscale).
    """
    print(f"\n{'='*60}")
    print(f"[VIS] Computing bilateral blur / vis control video")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Bilateral: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
    print(f"  Downscale factor: {downscale_factor}x")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale
        small_h, small_w = height // downscale_factor, width // downscale_factor
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Apply bilateral filter
        blurred = cv2.bilateralFilter(small, d, sigma_color, sigma_space)

        # Upscale back with bicubic
        result = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_CUBIC)

        out.write(result)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / frame_idx) * (total_frames - frame_idx)
            print(f"  Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    cap.release()
    out.release()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done! {frame_idx} frames in {elapsed:.1f}s | {size_mb:.1f} MB -> {output_path}")


# ============================================================================
# 3. DEPTH — Video-Depth-Anything (uses pre-downloaded model if available)
# ============================================================================


def compute_depth_video(input_path: str, output_path: str, encoder: str = "vits"):
    """Compute depth video using Video-Depth-Anything.

    Args:
        input_path: Path to input RGB video.
        output_path: Path to save depth video (.mp4).
        encoder: "vits" (small, fast) or "vitl" (large, slower but more accurate).
    """
    print(f"\n{'='*60}")
    print(f"[DEPTH] Computing depth video using Video-Depth-Anything ({encoder})")
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    print(f"  Encoder: {encoder} ({'small, fast' if encoder == 'vits' else 'large, accurate'})")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(ROOT / "cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py"),
        "--input_video", input_path,
        "--output_video", output_path,
        "--encoder", encoder,
    ]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ERROR: Depth pipeline failed with return code {result.returncode}")
        sys.exit(result.returncode)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done! {size_mb:.1f} MB -> {output_path}")


# ============================================================================
# 4. SEG — SAM2 + GroundingDINO (heavy, downloads ~10GB models)
# ============================================================================


def compute_seg_video(input_path: str, output_path: str, prompt: str = "robot arm gripper test tube rack"):
    """Compute segmentation video using SAM2 + GroundingDINO.

    Args:
        input_path: Path to input RGB video.
        output_path: Path to save segmentation video (.mp4).
        prompt: Text prompt for GroundingDINO to detect objects of interest.
    """
    print(f"\n{'='*60}")
    print(f"[SEG] Computing segmentation video using SAM2 + GroundingDINO")
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    print(f"  Prompt:  '{prompt}'")
    print(f"  NOTE: Downloads ~10GB of models (SAM2 + GroundingDINO) on first run")
    print(f"{'='*60}")

    # Temporary directory for SAM2 frame extraction
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="sam2_seg_")

    try:
        cmd = [
            sys.executable,
            str(ROOT / "cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py"),
            "--input_video", input_path,
            "--output_video", output_path,
            "--mode", "prompt",
            "--prompt", prompt,
        ]

        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  ERROR: SAM2 pipeline failed with return code {result.returncode}")
            sys.exit(result.returncode)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Done! {size_mb:.1f} MB -> {output_path}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="lerobot/seeed_rebot_b601_dm/organize_test_tube/videos/observation.images.side/chunk-000/file-000_25s.mp4",
        help="Path to input RGB video (relative to repo root, or absolute).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lerobot_controls_preview",
        help="Output directory for all control videos.",
    )
    parser.add_argument(
        "--edge", action="store_true",
        help="Compute edge (Canny) control video.",
    )
    parser.add_argument(
        "--depth", action="store_true",
        help="Compute depth control video (Video-Depth-Anything).",
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Compute vis (bilateral blur) control video.",
    )
    parser.add_argument(
        "--seg", action="store_true",
        help="Compute seg (SAM2+GroundingDINO) control video.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Compute all 4 control videos (edge + depth + vis + seg).",
    )
    parser.add_argument(
        "--edge_t_lower", type=int, default=100,
        help="Canny lower threshold for edge detection. Lower = more edges (default: 100, medium preset).",
    )
    parser.add_argument(
        "--edge_t_upper", type=int, default=200,
        help="Canny upper threshold for edge detection. Higher = cleaner edges (default: 200, medium preset).",
    )
    parser.add_argument(
        "--seg_prompt",
        type=str,
        default="robot arm gripper test tube rack laboratory",
        help="Text prompt for segmentation model.",
    )
    parser.add_argument(
        "--depth_encoder",
        type=str,
        choices=["vits", "vitl"],
        default="vits",
        help="Video-Depth-Anything encoder: vits=small/fast, vitl=large/accurate.",
    )

    args = parser.parse_args()

    # Resolve input path
    input_video = Path(args.input_video)
    if not input_video.is_absolute():
        input_video = ROOT / input_video

    if not input_video.exists():
        print(f"ERROR: Input video not found: {input_video}")
        sys.exit(1)

    output_dir = ROOT / args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check video info
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n{'#'*60}")
    print(f"# Pre-compute Control Videos for Cosmos Transfer 2.5")
    print(f"#")
    print(f"# Input:  {input_video}")
    print(f"# Output: {output_dir}/")
    print(f"# Video:  {total_frames} frames @ {fps} FPS, {width}x{height}")
    print(f"{'#'*60}")

    # Edge
    if args.edge or args.all:
        t0 = time.time()
        out = output_dir / "edge.mp4"
        compute_edge_video(
            str(input_video),
            str(out),
            t_lower=args.edge_t_lower,
            t_upper=args.edge_t_upper,
        )

    # Vis (blur)
    if args.vis or args.all:
        t0 = time.time()
        out = output_dir / "vis.mp4"
        compute_vis_video(str(input_video), str(out))

    # Depth
    if args.depth or args.all:
        t0 = time.time()
        out = output_dir / "depth.mp4"
        compute_depth_video(str(input_video), str(out), encoder=args.depth_encoder)

    # Seg
    if args.seg or args.all:
        t0 = time.time()
        out = output_dir / "seg.mp4"
        compute_seg_video(str(input_video), str(out), prompt=args.seg_prompt)

    # Summary
    print(f"\n{'='*60}")
    print(f"[DONE] All control videos saved to: {output_dir}/")
    print(f"{'='*60}")
    for f in sorted(output_dir.iterdir()):
        if f.suffix == ".mp4":
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name:<20} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
