#!/usr/bin/env python3
"""Run and score VLegal tasks 2.4 and 2.5."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("2.4", "2.5"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter tùy chọn để nạp lên trên --model.")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt tùy chọn (mặc định không có, chỉ dùng nguyên văn field trong file test).",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--precision", choices=("bf16", "4bit"), default="4bit")
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def parse_task_2_4(text: str) -> str | None:
    normalized = strip_accents(text).strip().upper()
    match = re.search(r"\b(DUNG|SAI)\b", normalized)
    return {"DUNG": "Đúng", "SAI": "Sai"}.get(match.group(1)) if match else None


def parse_task_2_5(text: str) -> list[str] | None:
    labels = sorted(set(re.findall(r"(?<![A-Z])[A-D](?![A-Z])", text.upper())))
    return labels or None


def build_messages(task: str, row: dict[str, Any], system_prompt: str | None) -> list[dict[str, str]]:
    if task == "2.4":
        prompt = "\n\n".join(
            [row["instruction"], row["description"], row["court_judgement"]]
        )
    else:
        prompt = "\n\n".join(
            [row["instruction"], row["question"], row["answers"]]
        )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch không nhìn thấy CUDA GPU.")

    rows = load_rows(args.data)
    selected = rows[args.offset:] if args.limit is None else rows[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("Không có mẫu nào trong khoảng offset/limit đã chọn.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # Model đã lượng tử hoá sẵn (AWQ/GPTQ) mang theo quantization_config riêng trong
    # config.json — không truyền thêm BitsAndBytesConfig, và kernel AWQ yêu cầu fp16.
    config_path = args.model / "config.json"
    is_prequantized = False
    if config_path.exists():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        quant_method = (model_config.get("quantization_config") or {}).get("quant_method")
        is_prequantized = quant_method is not None
        if quant_method == "awq":
            print("Phát hiện model đã lượng tử hoá sẵn (AWQ) — dùng fp16 thay vì bf16.")

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": {"": 0},
        "torch_dtype": torch.float16 if is_prequantized else torch.bfloat16,
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
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    invalid = 0
    results = []
    for number, row in enumerate(selected, 1):
        encoded = tokenizer.apply_chat_template(
            build_messages(args.task, row, args.system_prompt),
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
        if args.task == "2.4":
            prediction = parse_task_2_4(raw)
            truth = str(row["ground_truth"]).strip().title()
        else:
            prediction = parse_task_2_5(raw)
            truth = sorted(set(str(label).upper() for label in row["ground_truth"]))
        is_correct = prediction == truth
        correct += int(is_correct)
        invalid += int(prediction is None)
        result = {
            "index": args.offset + number - 1,
            "ground_truth": truth,
            "prediction": prediction,
            "correct": is_correct,
            "raw_output": raw,
        }
        if "id" in row:
            result["id"] = row["id"]
        results.append(result)
        print(
            f"[{number:04d}/{len(selected):04d}] pred={prediction} "
            f"truth={truth} correct={is_correct}",
            flush=True,
        )

    accuracy = correct / len(selected)
    summary = {
        "task": args.task,
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "system_prompt": args.system_prompt,
        "precision": quant_method if is_prequantized else args.precision,
        "data": str(args.data),
        "num_examples": len(selected),
        "correct": correct,
        "invalid": invalid,
        "accuracy": accuracy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Accuracy: {correct}/{len(selected)} = {accuracy:.2%}")
    print(f"Invalid: {invalid}/{len(selected)}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
