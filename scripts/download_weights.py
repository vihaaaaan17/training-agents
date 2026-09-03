"""
Utility script to download trained model weights and LoRA adapters from Hugging Face Hub.
"""

import os
import argparse
from huggingface_hub import snapshot_download

def download_model(repo_id: str, local_dir: str, token: str = None):
    print(f"Downloading repository: {repo_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token or os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print(f"[SUCCESS] Downloaded to {local_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download fine-tuned agent weights from Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g. username/gemma-2-2b-it-pi-mono-sft or username/gemma-2-2b-it-pi-mono-adapter-lr1e4-r16-len2k)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./weights",
        help="Local directory to store downloaded weights (default: ./weights)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional if public or logged in via huggingface-cli)",
    )
    args = parser.parse_args()
    download_model(args.repo_id, args.local_dir, args.token)
