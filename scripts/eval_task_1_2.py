#!/usr/bin/env python3
"""Run and score VLegal task 1.2 multiple-choice classification."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList

ANSWER_LETTERS = "ABCDEF"
DEFAULT_SYSTEM_PROMPT = "Chỉ trả lời duy nhất một chữ cái A, B, C, D, E hoặc F."


class AllowOnlyLogitsProcessor(LogitsProcessor):
    """Ép token sinh ra chỉ được là 1 trong các token_id cho phép (constrained decoding)."""

    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter tùy chọn để nạp lên trên --model.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt ràng buộc định dạng đầu ra. Truyền chuỗi rỗng (\"\") để tắt hẳn.",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_false",
        dest="constrained",
        default=True,
        help="Tắt constrained decoding (mặc định BẬT: ép token đầu ra chỉ được là 1 trong A-F, max-new-tokens=1).",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=("bf16", "4bit"),
        default="bf16",
        help="BF16 cho baseline chất lượng; 4bit để tiết kiệm VRAM.",
    )
    return parser.parse_args()


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
    match = re.search(r"(?:ĐÁP ÁN|ANSWER)\s*(?:LÀ|IS|:)?\s*([A-F])\b", normalized)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([A-F])(?:\s|[.):-]|$)", normalized)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch không nhìn thấy CUDA GPU.")
    rows = load_rows(args.data)
    selected = rows[args.offset : args.offset + args.limit]
    if not selected:
        raise ValueError("Không có mẫu nào trong khoảng offset/limit đã chọn.")

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
            print("Phát hiện model đã lượng tử hoá sẵn (AWQ) — dùng fp16 thay vì bf16.")

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": {"": 0},
        "torch_dtype": torch.float16 if is_prequantized else torch.bfloat16,
        "attn_implementation": "eager",
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

    logits_processor = None
    max_new_tokens = args.max_new_tokens
    if args.constrained:
        allowed_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ANSWER_LETTERS]
        logits_processor = LogitsProcessorList([AllowOnlyLogitsProcessor(allowed_ids)])
        max_new_tokens = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    correct = 0
    for position, row in enumerate(selected, args.offset):
        user_prompt = "\n\n".join(
            [row["instruction"], row["question"], row["answers"]]
        )
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
            )
        raw_output = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        prediction = extract_answer(raw_output)
        truth = str(row["ground_truth"]).strip().upper()
        is_correct = prediction == truth
        correct += int(is_correct)
        result = {
            "index": position,
            "question": row["question"],
            "answers": row["answers"],
            "ground_truth": truth,
            "prediction": prediction,
            "correct": is_correct,
            "raw_output": raw_output,
        }
        results.append(result)
        print(
            f"[{len(results):02d}/{len(selected):02d}] "
            f"pred={prediction or 'INVALID'} truth={truth} correct={is_correct}"
        )

    accuracy = correct / len(results)
    summary = {
        "task": "1.2",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "system_prompt": args.system_prompt,
        "precision": quant_method if is_prequantized else args.precision,
        "data": str(args.data),
        "num_examples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Accuracy: {correct}/{len(results)} = {accuracy:.2%}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
