#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run LeRobot side-camera video augmentation using Cosmos Transfer 2.5 Edge control.
# Strictly follows the NVIDIA Cosmos Cookbook format for simulator-to-real augmentation.
#
# Usage:
#   # Single GPU
#   bash scripts/run_lerobot_edge_augment.sh
#
#   # Multi-GPU (e.g., 4 GPUs)
#   bash scripts/run_lerobot_edge_augment.sh --nproc_per_node 4 --master_port 12341
#
# Output:
#   outputs/lerobot_edge_augment/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NUM_GPUS="${1:-1}"
PORT="${2:-12341}"

SPEC_FILE="assets/lerobot_example/edge/lerobot_edge_spec.json"
OUTPUT_DIR="outputs/lerobot_edge_augment"

echo "=============================================="
echo "Cosmos Transfer 2.5 — LeRobot Edge Augmentation"
echo "=============================================="
echo "Spec:      $SPEC_FILE"
echo "Output:    $OUTPUT_DIR"
echo "GPUs:      $NUM_GPUS"
echo "Port:      $PORT"
echo "=============================================="

python examples/inference.py \
    -i "$SPEC_FILE" \
    -o "$OUTPUT_DIR"
