#!/usr/bin/env python3
"""Chấm điểm model trên bộ test task 2.5 v3 (định dạng chat, không phải benchmark A-F).

Prompt lấy nguyên văn messages[0] (system) + messages[1] (user) trong file test.
Gold parse từ messages[2] (assistant). Model được yêu cầu sinh JSON {"intents":[...]}.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--precision", choices=("bf16", "4bit"), default="4bit")
    return p.parse_args()


def extract_intents(text: str) -> list[str] | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        obj = json.loads(candidate)
        intents = obj.get("intents")
        if isinstance(intents, list):
            return sorted(set(str(x) for x in intents))
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy CUDA GPU.")

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if args.precision == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
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
    tp = fp = fn = 0
    correct = 0
    invalid = 0
    for i, row in enumerate(rows, 1):
        messages = [
            {"role": "system", "content": row["messages"][0]["content"]},
            {"role": "user", "content": row["messages"][1]["content"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(gen[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        pred = extract_intents(raw)
        gold = sorted(set(json.loads(row["messages"][2]["content"])["intents"]))
        g, p = set(gold), set(pred or [])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        is_correct = pred is not None and p == g
        correct += int(is_correct)
        invalid += int(pred is None)
        results.append({
            "index": i - 1, "id": row.get("id"), "ground_truth": gold,
            "prediction": pred, "correct": is_correct, "raw_output": raw,
        })
        print(f"[{i:04d}/{len(rows):04d}] pred={pred} gold={gold} correct={is_correct}", flush=True)

    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    avg_pred = sum(len(r["prediction"] or []) for r in results) / len(results)
    avg_gold = sum(len(r["ground_truth"]) for r in results) / len(results)

    summary = {
        "model": str(args.model), "adapter": str(args.adapter) if args.adapter else None,
        "precision": args.precision, "data": str(args.data), "num_examples": len(rows),
        "exact_match": correct / len(rows), "invalid": invalid,
        "micro_precision": P, "micro_recall": R, "micro_f1": F1,
        "avg_pred_labels": avg_pred, "avg_gold_labels": avg_gold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nExact Match: {correct}/{len(rows)} = {correct/len(rows):.2%}")
    print(f"Micro P={P:.1%} R={R:.1%} F1={F1:.1%}  | avg_pred={avg_pred:.2f} avg_gold={avg_gold:.2f}")
    print(f"Invalid: {invalid}/{len(rows)}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
