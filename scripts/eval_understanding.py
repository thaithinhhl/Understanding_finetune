#!/usr/bin/env python3
"""Sinh prediction cho các task understanding (C1/U1/U2/U3).

Chỉ chạy inference và lưu output thô. Việc chấm điểm do scripts/score_understanding.py
đảm nhiệm theo metrics_understanding.md, để chỉnh metric không phải chạy lại GPU.

Prompt lấy nguyên văn messages[0] (system) + messages[1] (user) từ file test.
messages[2] (assistant) là gold, chỉ lưu lại để chấm, không đưa vào prompt.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

UNDERSTANDING_GROUPS = {"C1", "U1", "U2", "U3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter tùy chọn để nạp lên trên --model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--precision", choices=("bf16", "4bit"), default="4bit")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("group") in UNDERSTANDING_GROUPS:
                rows.append(row)
    return rows


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch không nhìn thấy CUDA GPU.")

    rows = load_rows(args.data)
    selected = rows[args.offset:] if args.limit is None else rows[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("Không có mẫu nào trong khoảng offset/limit đã chọn (kiểm tra group C1/U1/U2/U3).")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # Model đã lượng tử hoá sẵn (AWQ/GPTQ) mang theo quantization_config riêng trong
    # config.json — không truyền thêm BitsAndBytesConfig, và kernel AWQ (autoawq/Triton)
    # yêu cầu fp16, không tương thích bf16 (lỗi "Both operands must be same dtype").
    config_path = args.model / "config.json"
    is_prequantized = False
    if config_path.exists():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        quant_method = (model_config.get("quantization_config") or {}).get("quant_method")
        is_prequantized = quant_method is not None
        if quant_method == "awq":
            print(f"Phát hiện model đã lượng tử hoá sẵn (AWQ) — dùng fp16 thay vì bf16.")

    dtype = torch.float16 if is_prequantized else torch.bfloat16
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": {"": 0},
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }
    if args.precision == "4bit" and not is_prequantized:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for number, row in enumerate(selected, 1):
        messages = [
            {"role": "system", "content": row["messages"][0]["content"]},
            {"role": "user", "content": row["messages"][1]["content"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        prediction = extract_json(raw)
        results.append({
            "index": args.offset + number - 1,
            "id": row["id"],
            "group": row["group"],
            "task_name": row["task_name"],
            "slice": row["slice"],
            "gold": extract_json(row["messages"][2]["content"]),
            "prediction": prediction,
            "raw_output": raw,
        })
        print(
            f"[{number:04d}/{len(selected):04d}] group={row['group']} "
            f"json_ok={prediction is not None}",
            flush=True,
        )

    payload = {
        "task_family": "understanding",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": quant_method if is_prequantized else args.precision,
        "data": str(args.data),
        "max_new_tokens": args.max_new_tokens,
        "num_examples": len(results),
        "json_valid_rate": sum(1 for r in results if r["prediction"] is not None) / len(results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"JSON valid rate: {payload['json_valid_rate']:.1%}")
    print(f"Predictions: {args.output}")
    print("Chấm điểm bằng: scripts/score_understanding.py")


if __name__ == "__main__":
    main()
