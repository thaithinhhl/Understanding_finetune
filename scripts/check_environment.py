#!/usr/bin/env python3
"""Read-only preflight checks before a QLoRA run."""

import argparse
import shutil
import sys
from pathlib import Path


def gib(value: int) -> float:
    return value / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch chưa được cài.")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime (PyTorch): {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: PyTorch không nhìn thấy GPU CUDA.")
        return 1

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        print(
            f"GPU {index}: {props.name}; VRAM free/total: "
            f"{gib(free):.2f}/{gib(total):.2f} GiB; "
            f"BF16: {torch.cuda.is_bf16_supported()}"
        )

    disk = shutil.disk_usage(args.workspace)
    print(f"Disk free at {args.workspace}: {gib(disk.free):.2f} GiB")
    if args.model_path:
        if not args.model_path.exists():
            print(f"ERROR: Không tìm thấy model: {args.model_path}")
            return 1
        required = ["config.json", "tokenizer_config.json"]
        missing = [name for name in required if not (args.model_path / name).exists()]
        weights = list(args.model_path.glob("*.safetensors"))
        if missing or not weights:
            print(f"ERROR: Model thiếu file: {missing}; safetensors={len(weights)}")
            return 1
        print(f"Model files OK: {args.model_path} ({len(weights)} safetensors files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

