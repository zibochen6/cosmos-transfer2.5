#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run LeRobot side-camera video augmentation using Cosmos Transfer 2.5 multi-control
# (depth + edge + seg + vis). This follows the NVIDIA Cosmos Cookbook "multicontrol"
# approach, loading all 4 control branches simultaneously.
#
# Usage:
#   # Single GPU
#   bash scripts/run_lerobot_multicontrol_augment.sh
#
#   # Multi-GPU (e.g., 4 GPUs)
#   torchrun --nproc_per_node=4 --master_port=12341 examples/inference.py \
#       -i assets/lerobot_example/multicontrol/lerobot_multicontrol_spec.json \
#       -o outputs/lerobot_multicontrol_augment
#
# Output:
#   outputs/lerobot_multicontrol_augment/
#
# Note: depth control video (outputs/depth/robot_depth.mp4) has 121 frames.
# For full 750-frame output, regenerate depth with: python scripts/compute_lerobot_depth.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SPEC_FILE="assets/lerobot_example/multicontrol/lerobot_multicontrol_spec.json"
OUTPUT_DIR="outputs/lerobot_multicontrol_augment"

echo "=============================================="
echo "Cosmos Transfer 2.5 — LeRobot Multi-Control Augmentation"
echo "=============================================="
echo "Spec:      $SPEC_FILE"
echo "Output:    $OUTPUT_DIR"
echo "Controls:  depth + edge + seg + vis"
echo "=============================================="

python examples/inference.py \
    -i "$SPEC_FILE" \
    -o "$OUTPUT_DIR"
