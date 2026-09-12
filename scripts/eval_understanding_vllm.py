#!/usr/bin/env python3
"""Sinh prediction cho các task understanding (C1/U1/U2/U3) bằng vLLM.

Thay thế scripts/eval_understanding.py khi cần tốc độ: vLLM batch toàn bộ
prompt cùng lúc (continuous batching + kernel AWQ tối ưu hơn autoawq), thay vì
gọi model.generate() từng mẫu một như bản HF Transformers.

Output cùng schema với eval_understanding.py để scripts/score_understanding.py
dùng lại được không cần sửa. Hỗ trợ LoRA adapter qua --adapter (dùng
vllm.LoRARequest, không cần merge adapter vào base model).
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
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
    rows = load_rows(args.data)
    selected = rows[args.offset:] if args.limit is None else rows[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("Không có mẫu nào trong khoảng offset/limit đã chọn (kiểm tra group C1/U1/U2/U3).")

    config_path = args.model / "config.json"
    quant_method = None
    if config_path.exists():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        quant_method = (model_config.get("quantization_config") or {}).get("quant_method")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row in selected:
        messages = [
            {"role": "system", "content": row["messages"][0]["content"]},
            {"role": "user", "content": row["messages"][1]["content"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    lora_request = None
    llm_kwargs: dict[str, Any] = {}
    if args.adapter is not None:
        adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        lora_rank = int(adapter_config.get("r", 16))
        llm_kwargs.update(enable_lora=True, max_lora_rank=lora_rank, max_loras=1)
        lora_request = LoRARequest("legal-adapter", 1, str(args.adapter))

    llm = LLM(
        model=str(args.model),
        dtype="float16" if quant_method == "awq" else "bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **llm_kwargs,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for number, (row, out) in enumerate(zip(selected, outputs), 1):
        raw = out.outputs[0].text.strip()
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

    payload = {
        "task_family": "understanding",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": quant_method or "bf16",
        "backend": "vllm",
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
