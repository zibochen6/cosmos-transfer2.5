# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Download all models required for Cosmos-Transfer2.5 inference.

Usage:
    # Download core models only (transfer + base + tokenizers)
    python scripts/download_models.py

    # Download core + auxiliary models (Depth Anything, SAM2, GroundingDINO, Guardrail)
    python scripts/download_models.py --with-aux

    # Download everything including experimental
    python scripts/download_models.py --with-experimental

    # Download specific category only
    python scripts/download_models.py --category transfer
    python scripts/download_models.py --category base
    python scripts/download_models.py --category tokenizer
    python scripts/download_models.py --category aux
    python scripts/download_models.py --category experimental

    # List available models without downloading
    python scripts/download_models.py --list
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import the packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


AUX_MODELS = [
    {
        "name": "Video-Depth-Anything-Small",
        "repo_id": "depth-anything/Video-Depth-Anything-Small",
        "filename": "video_depth_anything_vits.pth",
        "description": "Depth estimation (small, fast)",
    },
    {
        "name": "Video-Depth-Anything-Large",
        "repo_id": "depth-anything/Video-Depth-Anything-Large",
        "filename": "video_depth_anything_vitl.pth",
        "description": "Depth estimation (large, high quality)",
    },
    {
        "name": "SAM2 Hiera Large",
        "repo_id": "facebook/sam2-hiera-large",
        "filename": None,
        "description": "Video object segmentation",
    },
    {
        "name": "GroundingDINO Base",
        "repo_id": "IDEA-Research/grounding-dino-base",
        "filename": None,
        "description": "Text-based object detection for SAM2",
    },
    {
        "name": "Cosmos-Guardrail1",
        "repo_id": "nvidia/Cosmos-Guardrail1",
        "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
        "filename": None,
        "description": "Content safety (text + video)",
    },
]

CHECKPOINT_CATEGORIES = {
    "tokenizer": [
        ("7219c6c7-f878-4137-bbdb-76842ea85e70", "Qwen/Qwen2.5-VL-7B-Instruct", "nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423", None),
        ("685afcaa-4de2-42fe-b7b9-69f7a2dee4d8", "Wan2.1 VAE", "nvidia/Cosmos-Predict2.5-2B", "f176dc95b4a70f53ce01c4b302851595e7322b00", "tokenizer.pth"),
    ],
    "base": [
        ("d20b7120-df3e-4911-919d-db6e08bad31c", "Cosmos-Predict2.5-2B base pre-trained", "nvidia/Cosmos-Predict2.5-2B", "15a82a2ec231bc318692aa0456a36537c806e7d4", "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"),
        ("81edfebe-bd6a-4039-8c1d-737df1a790bf", "Cosmos-Predict2.5-2B base post-trained", "nvidia/Cosmos-Predict2.5-2B", "15a82a2ec231bc318692aa0456a36537c806e7d4", "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"),
        ("54937b8c-29de-4f04-862c-e67b04ec41e8", "Cosmos-Predict2.5-14B base pre-trained", "nvidia/Cosmos-Predict2.5-14B", "03eb354f35eae0d6e0c1be3c9f94d8551e125570", "base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt"),
        ("e21d2a49-4747-44c8-ba44-9f6f9243715f", "Cosmos-Predict2.5-14B base post-trained", "nvidia/Cosmos-Predict2.5-14B", "2bc4ca5ba5a20b9858a7ddb856bc82d70b030fbe", "base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"),
    ],
    "transfer": [
        ("61f5694b-0ad5-4ecd-8ad7-c8545627d125", "Edge Control (rectified flow, 720p)", "nvidia/Cosmos-Transfer2.5-2B", "b67b64abda3801a9aceddbff2bdb86126c06db74", "general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt"),
        ("626e6618-bfcd-4d9a-a077-1409e2ce353f", "Depth Control (rectified flow, 720p)", "nvidia/Cosmos-Transfer2.5-2B", "dea7737ca29dd8d9086413c6dc5724b8250a0bb4", "general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt"),
        ("ba2f44f2-c726-4fe7-949f-597069d9b91c", "Visual/Blur Control (rectified flow, 720p)", "nvidia/Cosmos-Transfer2.5-2B", "eb5325b77d358944da58a690157dd2b8071bbf85", "general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt"),
        ("5136ef49-6d8d-42e8-8abf-7dac722a304a", "Segmentation Control (rectified flow, 720p)", "nvidia/Cosmos-Transfer2.5-2B", "23057a4167b89de89a4a397fdbf3887994d115eb", "general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt"),
        ("4ecc66e9-df19-4aed-9802-0d11e057287a", "Auto Multiview (7 views, 720p)", "nvidia/Cosmos-Transfer2.5-2B", "00c591edab119e8a6ca06e6e091351a04ce0ecc9", "auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt"),
        ("41f07f13-f2e4-4e34-ba4c-86f595acbc20", "Distilled Edge (low-latency)", "nvidia/Cosmos-Transfer2.5-2B", "bbaeedb2b57cc8b14a44653099e2551adb69dcc7", "distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt"),
    ],
    "experimental": [
        ("ecd0ba00-d598-4f94-aa09-e8627899c431", "Edge Control (non-uniform)", "nvidia/Cosmos-Transfer2.5-2B", "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5", "general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt"),
        ("fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab", "Seg Control (non-uniform)", "nvidia/Cosmos-Transfer2.5-2B", "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5", "general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt"),
        ("20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1", "Blur Control (non-uniform)", "nvidia/Cosmos-Transfer2.5-2B", "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5", "general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt"),
        ("0f214f66-ae98-43cf-ab25-d65d09a7e68f", "Depth Control (non-uniform)", "nvidia/Cosmos-Transfer2.5-2B", "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5", "general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt"),
        ("b5ab002d-a120-4fbf-a7f9-04af8615710b", "Auto Multiview (alternate)", "nvidia/Cosmos-Transfer2.5-2B", "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5", "auto/multiview/b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt"),
        ("0e8177cc-0db5-4cfd-a8a4-b820c772f4fc", "Robot Multiview (multi-camera)", "nvidia/Cosmos-Experimental", "9a02ed8daa8c6c7718ac09da06488bfd1d363cb6", "0e8177cc-0db5-4cfd-a8a4-b820c772f4fc/model_ema_bf16.pt"),
        ("7f6b99b7-7fac-4e74-8dbe-a394cb56ef99", "Robot Multiview (agibot)", "nvidia/Cosmos-Experimental", "9a02ed8daa8c6c7718ac09da06488bfd1d363cb6", "7f6b99b7-7fac-4e74-8dbe-a394cb56ef99/model_ema_bf16.pt"),
        ("a8794d70-842c-44a5-95bb-9010d5ace7be", "Robot Multiview (many-camera, 4in-1out)", "nvidia/Cosmos-Experimental", "main", "a8794d70-842c-44a5-95bb-9010d5ace7be/model_ema_bf16.pt"),
        ("32514ba1-6d05-4ce5-997d-a3b5bf894cab", "Robot Multiview Agibot Depth Control", "nvidia/Cosmos-Experimental", "main", "32514ba1-6d05-4ce5-997d-a3b5bf894cab/model_ema_bf16.pt"),
        ("fffbd388-89c9-4604-ad4f-6c6b36272c48", "Robot Multiview Agibot Edge Control", "nvidia/Cosmos-Experimental", "main", "fffbd388-89c9-4604-ad4f-6c6b36272c48/model_ema_bf16.pt"),
        ("2eca9f80-bf8f-4257-b05f-278065d21500", "Robot Multiview Agibot Vis Control", "nvidia/Cosmos-Experimental", "main", "2eca9f80-bf8f-4257-b05f-278065d21500/model_ema_bf16.pt"),
        ("c5a9a58b-7f3e-4b45-9e5d-8f7b3d4e5a6c", "Robot Multiview Agibot Seg Control", "nvidia/Cosmos-Experimental", "main", "c5a9a58b-7f3e-4b45-9e5d-8f7b3d4e5a6c/model_ema_bf16.pt"),
        ("308eb96c-c4c0-4a06-9cc1-103a43beff28", "Cosmos-Predict2.5-2B (alternate)", "nvidia/Cosmos-Experimental", "eda2f0ca1db6281c9a960908bb6bf14607a0fea0", "308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt"),
        ("7bbc8d06-2bc9-448d-94ee-b48b4ab7189c", "Cosmos-Predict2.5-2B action-conditioned", "nvidia/Cosmos-Experimental", "2b5e9a99b58d5a61259ca99962c4c74127481006", "7bbc8d06-2bc9-448d-94ee-b48b4ab7189c/model_ema_bf16.pt"),
        ("bedc35da-1a54-4144-83db-6072c29b0fd9", "Cosmos-Predict2.5-2B action (warmup)", "nvidia/Cosmos-Experimental", "ded876a5b2e19aef64cd9d1100c03e5b05cf2f9c", "bedc35da-1a54-4144-83db-6072c29b0fd9/model_ema_bf16.pt"),
        ("524af350-2e43-496c-8590-3646ae1325da", "Predict2.5 Multiview (7 views)", "nvidia/Cosmos-Predict2.5-2B", "865baf084d4c9e850eac59a021277d5a9b9e8b63", "auto/multiview/524af350-2e43-496c-8590-3646ae1325da_ema_bf16.pt"),
        ("6b9d7548-33bb-4517-b5e8-60caf47edba7", "Predict2.5 Multiview (alternate)", "nvidia/Cosmos-Predict2.5-2B", "15a82a2ec231bc318692aa0456a36537c806e7d4", "auto/multiview/6b9d7548-33bb-4517-b5e8-60caf47edba7_ema_bf16.pt"),
        ("f740321e-2cd6-4370-bbfe-545f4eca2065", "Predict2.5 Robot Multiview (agibot frameinit)", "nvidia/Cosmos-Predict2.5-2B", "fbe72c18d152053029a19db3b211cf78671ad422", "robot/multiview-agibot/f740321e-2cd6-4370-bbfe-545f4eca2065_ema_bf16.pt"),
        ("38c6c645-7d41-4560-8eeb-6f4ddc0e6574", "Predict2.5 Robot Action-Conditioned", "nvidia/Cosmos-Predict2.5-2B", "main", "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"),
        ("24a3b7b8-6a3d-432d-b7d1-5d30b9229465", "Cosmos-Predict2.5-2B (transfer2.5 variant)", "nvidia/Cosmos-Experimental", "9a02ed8daa8c6c7718ac09da06488bfd1d363cb6", "24a3b7b8-6a3d-432d-b7d1-5d30b9229465/model_ema_bf16.pt"),
        ("575edf0f-d973-4c74-b52c-69929a08d0a5", "Cosmos-Predict2.5-2B distilled", "nvidia/Cosmos-Predict2.5-2B", "e26f8a125a2235c5a00245a65207402dd0cdcb89", "base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt"),
        ("cb3e3ffa-7b08-4c34-822d-61c7aa31a14f", "Cosmos-Reason1.1-7B (VQA)", "nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423", None),
    ],
}


def _format_size(path: Path) -> str:
    """Return human-readable size of a file or directory."""
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _download_hf_file(repo_id: str, filename: str | None, revision: str | None = None) -> Path:
    """Download a file or entire repo from HuggingFace using the Python API."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    if filename:
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir)
        return Path(path)
    else:
        path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
        return Path(path)


def _download_hf_model(repo_id: str, filename: str | None, revision: str | None = None) -> Path:
    """Download a model (or specific file) from HuggingFace using the Python API."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    if filename:
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir)
        return Path(path)
    else:
        path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
        return Path(path)


def _list_models(category: str | None, with_aux: bool, with_experimental: bool) -> list[str]:
    """Return a flat list of model identifiers for the given category."""
    lines = []
    categories = []
    if category:
        if category == "aux":
            categories = []
        else:
            categories = [category]
    else:
        categories = ["tokenizer", "base", "transfer"]
        if with_experimental:
            categories.append("experimental")

    for cat in categories:
        for entry in CHECKPOINT_CATEGORIES.get(cat, []):
            uuid, name, *_ = entry
            lines.append(f"  [{cat.upper()}] {name} ({uuid})")

    if with_aux or (category == "aux"):
        for m in AUX_MODELS:
            lines.append(f"  [AUX]   {m['name']} ({m['repo_id']})")

    return lines


def run(args: argparse.Namespace) -> int:
    # List mode
    if args.list:
        print("Available models:\n")
        for line in _list_models(args.category, args.with_aux, args.with_experimental):
            print(line)
        print()
        print(f"Cache location: {os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')}")
        return 0

    # Determine which categories to download
    categories_to_download = ["tokenizer", "base", "transfer"]
    if args.with_experimental or args.category == "experimental":
        categories_to_download.append("experimental")

    if args.category:
        categories_to_download = [args.category]

    failed = []

    # Download checkpoint-based models using the Python API directly
    for cat in categories_to_download:
        entries = CHECKPOINT_CATEGORIES.get(cat, [])
        if not entries:
            continue
        print(f"\n[{cat.upper()}] Downloading {len(entries)} checkpoint(s)...")

        for entry in entries:
            uuid, name, repo_id, revision, filename = entry
            print(f"  Downloading {name} ({repo_id})...", end=" ", flush=True)
            try:
                path = _download_hf_file(repo_id, filename, revision)
                size = _format_size(path)
                print(f"-> {path} ({size})")
            except Exception as e:
                print(f"[FAIL] {e}")
                failed.append((cat, name, uuid))

    # Download auxiliary models
    if args.with_aux or args.category == "aux":
        print(f"\n[AUX] Downloading {len(AUX_MODELS)} auxiliary model(s)...")
        for m in AUX_MODELS:
            repo_id = m["repo_id"]
            filename = m.get("filename")
            revision = m.get("revision")
            name = m["name"]

            print(f"  Downloading {name} ({repo_id})...", end=" ", flush=True)
            try:
                path = _download_hf_model(repo_id, filename, revision)
                size = _format_size(path)
                print(f"-> {path} ({size})")
            except Exception as e:
                print(f"[FAIL] {e}")
                failed.append(("aux", name, repo_id))

    # Summary
    print()
    if failed:
        print(f"Download failed for {len(failed)} model(s):")
        for cat, name, identifier in failed:
            print(f"  [{cat.upper()}] {name} ({identifier})")
        return 1
    else:
        print("All models downloaded successfully.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download all models required for Cosmos-Transfer2.5 inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--with-aux",
        action="store_true",
        help="Also download auxiliary models (Depth Anything, SAM2, GroundingDINO, Guardrail)",
    )
    parser.add_argument(
        "--with-experimental",
        action="store_true",
        help="Also download experimental checkpoints",
    )
    parser.add_argument(
        "--category",
        choices=["tokenizer", "base", "transfer", "aux", "experimental"],
        help="Download only a specific category of models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models without downloading",
    )
    args = parser.parse_args()

    if args.with_experimental and args.category:
        parser.error("Cannot use --with-experimental together with --category")

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
