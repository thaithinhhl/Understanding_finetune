#!/usr/bin/env python3
"""Download a Hugging Face model into a stable project directory."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="Ví dụ: Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("local_dir", type=Path, help="Thư mục đích trên host")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        revision=args.revision,
    )
    print(f"Model downloaded to: {path}")


if __name__ == "__main__":
    main()

