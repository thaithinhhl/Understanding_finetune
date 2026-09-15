#!/usr/bin/env python3
"""Chấm benchmark task 2.5 (trắc nghiệm A-D đa nhãn) bằng vLLM, dùng đúng prompt
chung lưu trong prompt_gold_task2.5.json (tách System:/User:, thay {instruction}/
{question}/{answers}). Hỗ trợ nạp thêm LoRA adapter và model đã lượng tử hoá sẵn
(AWQ). Chấm Exact Match + Micro-P/R/F1 (đa nhãn)."""

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

# Mô tả 8 intent, lấy nguyên văn từ system prompt lúc train nhóm V3
# (data-finetune-v2/data_finetune_v2_train.jsonl) — dùng cho --train-prompt.
INTENT_DESCRIPTIONS = {
    "chitchat": "Câu hỏi không liên quan đến pháp luật (ví dụ chào hỏi, cảm ơn, off-topic)",
    "comparative_analysis": "So sánh nội dung giữa hai văn bản, điều khoản, nội dung, ...",
    "document_relationship": "Câu hỏi về mối quan hệ giữa các văn bản. ví dụ về sửa đổi, bổ sung - hướng dẫn - dẫn chiếu - căn cứ",
    "document_retrieval": "Truy xuất toàn văn bản pháp luật",
    "external_analysis": "Tác động kinh tế, xã hội, xu hướng thay đổi, lịch sử, tác động, ảnh hưởng, xu hướng.",
    "general": "Câu hỏi tổng quát, có nội dung liên quan đến pháp luật, chưa thuộc intent nào cụ thể",
    "legal_query": "Tìm và trả lời từ nội dung cụ thể của điều / khoản / mục / điểm cụ thể",
    "stats_summary": "Thống kê số lượng văn bản/quy định.",
}

TRAIN_SYSTEM_TEMPLATE = (
    "Đọc query sau và xác định đúng intent của câu hỏi đó. Chỉ trả lời bằng tên intent, "
    "không giải thích gì thêm. Danh sách các intent: \n{intent_list}\n\n"
    'Chỉ trả về JSON: {{"intents": ["legal_query"]}} — một hoặc nhiều tên intent.'
)

MULTI_LABEL_HINT = (
    " Câu hỏi có thể phù hợp với NHIỀU intent cùng lúc — hãy chọn đầy đủ tất cả các intent "
    "đúng, không chỉ chọn một."
)

ALL_INTENTS = set(INTENT_DESCRIPTIONS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter tùy chọn để nạp lên trên --model.")
    p.add_argument("--prompt-file", type=Path, default=Path("prompt_gold_task2.5.json"))
    p.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Bỏ qua --prompt-file, dùng nguyên instruction+question+answers từ benchmark "
        "làm user message, không thêm system prompt nào.",
    )
    p.add_argument(
        "--train-prompt",
        action="store_true",
        help="Dùng đúng system prompt lúc train nhóm V3 (data-finetune-v2). Model sinh JSON "
        '{"intents": [...]}, map ngược về chữ cái. Mặc định liệt kê đủ cả 8 intent như lúc '
        "train thật; thêm --narrow-intents để thu hẹp còn đúng 4 intent có mặt trong lựa chọn "
        "của câu hỏi benchmark (biến thể so sánh).",
    )
    p.add_argument(
        "--narrow-intents",
        action="store_true",
        help="Chỉ có tác dụng cùng --train-prompt: thu hẹp danh sách intent trong system "
        "prompt xuống còn đúng 4 lựa chọn của câu hỏi benchmark thay vì đủ 8 intent lúc train.",
    )
    p.add_argument(
        "--multi-label-hint",
        action="store_true",
        help="Chỉ có tác dụng cùng --train-prompt: thêm 1 câu nhắc ngắn vào cuối system "
        "prompt rằng câu hỏi có thể phù hợp với nhiều intent cùng lúc, hãy chọn đầy đủ.",
    )
    p.add_argument(
        "--score-intents",
        action="store_true",
        help="Với --train-prompt, đổi ground-truth A-D sang tên intent và chấm trực tiếp "
        "trong không gian 8 intent. Intent dự đoán ngoài 4 lựa chọn vẫn là false positive.",
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=16)
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


def extract_answers(text: str, valid_letters: set[str]) -> set[str]:
    normalized = text.strip().upper()
    letters = set(re.findall(r"\b([A-D])\b", normalized))
    return letters & valid_letters


def parse_letter_to_intent(answers: str) -> dict[str, str]:
    return dict(re.findall(r"([A-D])\.\s*(\S+)", answers.strip()))


def extract_intents(raw: str, valid_intents: set[str]) -> set[str]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            intents = obj.get("intents")
            if isinstance(intents, list):
                found = {str(i).strip() for i in intents}
                return found & valid_intents
        except json.JSONDecodeError:
            pass
    # fallback: quét trực tiếp tên intent xuất hiện trong văn bản thô
    return {intent for intent in valid_intents if re.search(rf"\b{re.escape(intent)}\b", text)}


def main() -> None:
    args = parse_args()
    if args.raw_prompt and args.train_prompt:
        raise SystemExit("Chỉ được chọn một trong --raw-prompt hoặc --train-prompt.")
    if args.score_intents and not args.train_prompt:
        raise SystemExit("--score-intents chỉ dùng cùng --train-prompt.")
    mode = "raw" if args.raw_prompt else ("train" if args.train_prompt else "prompt_gold")

    system_prompt = None
    user_template = None
    if mode == "prompt_gold":
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
        if mode == "raw":
            user_content = "\n\n".join([row["instruction"], row["question"], row["answers"]])
            messages = [{"role": "user", "content": user_content}]
        elif mode == "train":
            listed_intents = (
                list(parse_letter_to_intent(row["answers"]).values())
                if args.narrow_intents
                else list(INTENT_DESCRIPTIONS)
            )
            intent_list = "\n".join(f"- {intent}: {INTENT_DESCRIPTIONS[intent]}" for intent in listed_intents)
            system_content = TRAIN_SYSTEM_TEMPLATE.format(intent_list=intent_list)
            if args.multi_label_hint:
                system_content += MULTI_LABEL_HINT
            user_content = f"Earlier turns:\n(none)\n\nCurrent turn:\n{row['question']}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        else:
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
    exact_matches = 0
    tp = fp = fn = 0
    total_pred = 0
    total_gold = 0
    for row, out in zip(selected, outputs):
        raw_output = out.outputs[0].text.strip()
        if mode == "train":
            letter_to_intent = parse_letter_to_intent(row["answers"])
            intent_to_letter = {v: k for k, v in letter_to_intent.items()}
            # Nhận diện trên toàn bộ 8 intent (model có thể sinh bất kỳ tên nào trong số
            # đã liệt kê ở system prompt), nhưng chỉ intent nào trùng với 1 trong 4 lựa
            # chọn của câu hỏi mới map được sang chữ cái để chấm điểm.
            predicted_intents = extract_intents(raw_output, ALL_INTENTS)
            if args.score_intents:
                prediction = predicted_intents
            else:
                prediction = {intent_to_letter[i] for i in predicted_intents if i in intent_to_letter}
        else:
            valid_letters = set(re.findall(r"\b([A-D])\.\s", row["answers"]))
            prediction = extract_answers(raw_output, valid_letters)
        gold_letters = {str(letter).strip().upper() for letter in row["ground_truth"]}
        gold = (
            {letter_to_intent[letter] for letter in gold_letters if letter in letter_to_intent}
            if mode == "train" and args.score_intents
            else gold_letters
        )

        is_exact = prediction == gold
        exact_matches += int(is_exact)
        tp += len(prediction & gold)
        fp += len(prediction - gold)
        fn += len(gold - prediction)
        total_pred += len(prediction)
        total_gold += len(gold)

        results.append(
            {
                "question": row["question"][:200],
                "answers": row["answers"],
                "ground_truth": sorted(gold),
                "prediction": sorted(prediction),
                "exact_match": is_exact,
                "raw_output": raw_output,
            }
        )

    n = len(results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        "task": "2.5-prompt-gold-vllm",
        "model": args.label or str(args.model),
        "model_path": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision_mode": "awq" if is_prequantized else "bf16",
        "backend": "vllm",
        "prompt_mode": mode,
        "score_space": "intents" if args.score_intents else "letters",
        "prompt_file": {
            "raw": "raw (instruction+question+answers, no system prompt)",
            "train": (
                ("train (intents narrowed to benchmark's 4 choices)" if args.narrow_intents else "train (full 8-intent list as in actual training)")
                + (" + multi-label hint" if args.multi_label_hint else "")
            ),
            "prompt_gold": str(args.prompt_file),
        }[mode],
        "data": str(args.data),
        "num_examples": n,
        "exact_match": exact_matches / n,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "avg_pred_per_example": total_pred / n,
        "avg_gold_per_example": total_gold / n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {summary['model']} ===")
    print(f"Exact Match : {exact_matches}/{n} = {exact_matches/n:.2%}")
    print(f"Micro-F1    : {micro_f1:.2%} (P={precision:.2%}, R={recall:.2%})")
    print(f"avg pred/gold per example: {total_pred/n:.2f} / {total_gold/n:.2f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
