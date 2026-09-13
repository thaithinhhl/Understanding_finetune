#!/usr/bin/env python3
"""Chấm task 1.2 dùng đúng prompt lưu trong prompt_gold_task1.2.json (tách System:/User:,
thay {question} và {choice_list})."""

import argparse
import json
import re
import unicodedata
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
    p.add_argument("--prompt-file", type=Path, default=Path("prompt_gold_task1.2.json"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--precision", choices=("bf16", "4bit"), default="bf16")
    return p.parse_args()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.strip().lower().split())


def parse_choices(answers: str) -> dict[str, str]:
    parts = re.findall(r"([A-F])\.\s*(.+?)(?=\s+[A-F]\.\s|$)", answers.strip())
    return {letter: norm(text) for letter, text in parts}


def extract_topic(raw: str) -> str | None:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            topic = obj.get("topic")
            if isinstance(topic, str) and topic.strip():
                return norm(topic)
        except json.JSONDecodeError:
            pass
    return norm(raw) if raw else None


def map_topic_to_letter(topic: str | None, choices: dict[str, str]) -> str | None:
    if topic is None:
        return None
    for letter, name in choices.items():
        if name == topic:
            return letter
    for letter, name in choices.items():
        if topic in name or name in topic:
            return letter
    return None


def main() -> None:
    args = parse_args()
    prompt_spec = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    full_prompt = prompt_spec["prompt"]
    _, rest = full_prompt.split("System:", 1)
    system_part, user_part = rest.split("User:", 1)
    system_prompt = system_part.strip()
    user_template = user_part.strip()

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8") if l.strip()]
    selected = rows[: args.limit] if args.limit else rows

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_kwargs: dict[str, Any] = {
        "local_files_only": True, "device_map": {"": 0},
        "torch_dtype": torch.bfloat16, "attn_implementation": "eager",
    }
    if args.precision == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    correct = 0
    unmapped = 0
    for i, row in enumerate(selected, 1):
        choices = parse_choices(row["answers"])
        raw_choices = re.findall(r"[A-F]\.\s*(.+?)(?=\s+[A-F]\.\s|$)", row["answers"].strip())
        choice_list = "\n".join(f"- {name.strip()}" for name in raw_choices)
        user_content = user_template.replace("{question}", row["question"]).replace("{choice_list}", choice_list)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        raw_output = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        topic = extract_topic(raw_output)
        prediction = map_topic_to_letter(topic, choices)
        truth = str(row["ground_truth"]).strip().upper()
        is_correct = prediction == truth
        correct += int(is_correct)
        unmapped += int(prediction is None)
        results.append({
            "index": i, "ground_truth": truth, "predicted_topic": topic,
            "prediction": prediction, "correct": is_correct, "raw_output": raw_output,
        })
        print(f"[{i:03d}/{len(selected):03d}] topic={topic!r} pred={prediction or 'UNMAPPED'} truth={truth} correct={is_correct}")

    accuracy = correct / len(results)
    summary = {
        "task": "1.2-prompt-gold", "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": args.precision, "prompt_file": str(args.prompt_file),
        "data": str(args.data), "num_examples": len(results), "correct": correct,
        "unmapped": unmapped, "accuracy": accuracy,
        "generated_at": datetime.now(timezone.utc).isoformat(), "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.2%}")
    print(f"Unmapped: {unmapped}/{len(results)}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
