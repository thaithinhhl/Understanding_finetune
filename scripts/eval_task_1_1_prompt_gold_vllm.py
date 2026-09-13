#!/usr/bin/env python3
"""Chấm benchmark task 1.1 (trắc nghiệm A-D) bằng vLLM, dùng đúng prompt chung
lưu trong prompt_gold_task1.1.json (tách System:/User:, thay {instruction}/
{question}/{answers}). Hỗ trợ nạp thêm LoRA adapter và model đã lượng tử hoá
sẵn (AWQ)."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ANSWER_LETTERS = "ABCD"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter tùy chọn để nạp lên trên --model.")
    p.add_argument("--prompt-file", type=Path, default=Path("prompt_gold_task1.1.json"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--label", default=None, help="Tên hiển thị trong summary (mặc định dùng --model).")
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"instruction", "question", "answers", "ground_truth"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"Dòng {line_number} thiếu field: {sorted(missing)}")
            rows.append(row)
    return rows


def extract_answer(text: str) -> str | None:
    normalized = text.strip().upper()
    match = re.search(r"(?:ĐÁP ÁN|ANSWER)\s*(?:LÀ|IS|:)?\s*([A-D])\b", normalized)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([A-D])(?:\s|[.):-]|$)", normalized)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()

    prompt_spec = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    full_prompt = prompt_spec["prompt"]
    _, rest = full_prompt.split("System:", 1)
    system_part, user_part = rest.split("User:", 1)
    system_prompt = system_part.strip()
    user_template = user_part.strip()

    rows = load_rows(args.data)
    selected = rows[: args.limit] if args.limit else rows

    is_prequantized = False
    config_path = args.model / "config.json"
    if config_path.exists():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        is_prequantized = (model_config.get("quantization_config") or {}).get("quant_method") is not None

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row in selected:
        user_content = (
            user_template.replace("{instruction}", row["instruction"])
            .replace("{question}", row["question"])
            .replace("{answers}", row["answers"])
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    llm_kwargs: dict[str, Any] = {}
    lora_request = None
    if args.adapter is not None:
        cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        llm_kwargs.update(enable_lora=True, max_lora_rank=int(cfg.get("r", 16)), max_loras=1)
        lora_request = LoRARequest("legal-adapter", 1, str(args.adapter))

    llm = LLM(
        model=str(args.model),
        dtype="float16" if is_prequantized else "bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        **llm_kwargs,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
        lora_request=lora_request,
    )

    results = []
    correct = 0
    for row, out in zip(selected, outputs):
        raw_output = out.outputs[0].text.strip()
        prediction = extract_answer(raw_output)
        truth = str(row["ground_truth"]).strip().upper()
        is_correct = prediction == truth
        correct += int(is_correct)
        results.append(
            {
                "question": row["question"][:200],
                "answers": row["answers"],
                "ground_truth": truth,
                "prediction": prediction,
                "correct": is_correct,
                "raw_output": raw_output,
            }
        )

    n = len(results)
    accuracy = correct / n
    summary = {
        "task": "1.1-prompt-gold-vllm",
        "model": args.label or str(args.model),
        "model_path": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": "awq" if is_prequantized else "bf16",
        "backend": "vllm",
        "prompt_file": str(args.prompt_file),
        "data": str(args.data),
        "num_examples": n,
        "correct": correct,
        "accuracy": accuracy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {summary['model']} ===")
    print(f"Accuracy: {correct}/{n} = {accuracy:.2%}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
