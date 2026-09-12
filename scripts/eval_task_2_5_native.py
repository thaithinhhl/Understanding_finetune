#!/usr/bin/env python3
"""Chấm benchmark task 2.5 bằng cách để model trả lời THEO ĐÚNG PROMPT LÚC TRAIN
(JSON tên intent), thay vì ép nó chọn trắc nghiệm A-D.

Lý do: adapter được train với system prompt V3 (liệt kê 8 intent, yêu cầu trả
JSON {"intents":[...]}). Ép nó sinh chữ cái A-D là format chưa từng thấy lúc
train nên không phản ánh đúng năng lực. Script này:

  1. Dùng nguyên văn system prompt V3 lấy từ file train (không hard-code).
  2. Bọc câu hỏi benchmark đúng format user lúc train
     ("Earlier turns:\\n(none)\\n\\nCurrent turn:\\n<question>").
  3. Parser đa tầng: JSON chuẩn -> tên intent trần -> chữ cái A-D -> văn xuôi.
  4. Map tên intent về chữ cái bằng chính bảng `answers` của từng câu, rồi chấm
     EM + Micro-P/R/F1 so với ground_truth.

Nhãn model sinh ra nằm ngoài 4 lựa chọn của câu đó được tính là FP — vì
ground_truth chắc chắn không chứa nó, tức đó là một dự đoán sai bình thường.
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

INTENTS = [
    "chitchat", "comparative_analysis", "document_relationship", "document_retrieval",
    "external_analysis", "general", "legal_query", "stats_summary",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True, help="Benchmark A-D (instruction/question/answers/ground_truth).")
    p.add_argument("--prompt-source", type=Path, default=Path("data-finetune-v2/data_finetune_v2_train.jsonl"),
                   help="File train để lấy nguyên văn system prompt của nhóm V3.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--nudge", action="store_true",
                   help="Thêm câu nhắc 'có thể nhiều intent, liệt kê TẤT CẢ' vào cuối system prompt "
                        "(chỉ để CHẨN ĐOÁN mức dè dặt của model, prompt sẽ lệch khỏi prompt lúc train).")
    return p.parse_args()


NUDGE = ("\n\nLưu ý: một câu hỏi có thể mang MỘT HOẶC NHIỀU intent cùng lúc. "
         "Hãy liệt kê TẤT CẢ intent phù hợp, không bỏ sót intent nào.")


def load_v3_system_prompt(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("group") == "V3":
                return row["messages"][0]["content"]
    raise SystemExit("Không tìm thấy mẫu V3 nào trong file prompt-source.")


def parse_choices(answers: str) -> dict[str, str]:
    """'A. legal_query B. comparative_analysis ...' -> {'A': 'legal_query', ...}"""
    return {letter: name for letter, name in re.findall(r"([A-D])\.\s*(\w+)", answers)}


def parse_prediction(raw: str, letter2intent: dict[str, str]) -> tuple[list[str], list[str], str]:
    """-> (tên intent hợp lệ, chữ cái sinh trực tiếp, tier parser đã dùng)"""
    text = raw.strip()

    # Tier 1: JSON chuẩn
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            intents = obj.get("intents")
            if isinstance(intents, list):
                names = [str(x).strip() for x in intents if str(x).strip() in INTENTS]
                if names:
                    return names, [], "json"
        except json.JSONDecodeError:
            pass

    # Tier 2: tên intent xuất hiện trong text (JSON hỏng, hoặc trả tên trần)
    names = [i for i in INTENTS if re.search(rf"\b{i}\b", text)]
    if names:
        return names, [], "names"

    # Tier 3: model lỡ trả chữ cái A-D
    letters = sorted(set(re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text.upper())))
    letters = [l for l in letters if l in letter2intent]
    if letters:
        return [], letters, "letters"

    return [], [], "invalid"


def main() -> None:
    args = parse_args()

    system_prompt = load_v3_system_prompt(args.prompt_source)
    if args.nudge:
        system_prompt += NUDGE
    rows = [json.loads(l) for l in args.data.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row in rows:
        user = f"Earlier turns:\n(none)\n\nCurrent turn:\n{row['question']}"
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        )

    llm_kwargs: dict[str, Any] = {}
    lora_request = None
    if args.adapter is not None:
        adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        llm_kwargs.update(enable_lora=True, max_lora_rank=int(adapter_config.get("r", 16)), max_loras=1)
        lora_request = LoRARequest("legal-adapter", 1, str(args.adapter))

    llm = LLM(
        model=str(args.model), dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **llm_kwargs,
    )
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
                           lora_request=lora_request)

    tp = fp = fn = 0            # intent ngoài 4 lựa chọn tính là FP
    exact = 0
    invalid = 0
    unmapped_total = 0
    rows_with_unmapped = 0
    tier_count: dict[str, int] = {}
    results = []

    for row, out in zip(rows, outputs):
        raw = out.outputs[0].text.strip()
        letter2intent = parse_choices(row["answers"])
        intent2letter = {v: k for k, v in letter2intent.items()}

        names, direct_letters, tier = parse_prediction(raw, letter2intent)
        tier_count[tier] = tier_count.get(tier, 0) + 1
        invalid += int(tier == "invalid")

        mapped = {intent2letter[n] for n in names if n in intent2letter}
        unmapped = [n for n in names if n not in intent2letter]
        unmapped_total += len(unmapped)
        rows_with_unmapped += int(bool(unmapped))
        pred_letters = mapped | set(direct_letters)

        gold = {str(g).strip().upper() for g in row["ground_truth"]}

        exact += int(pred_letters == gold and not unmapped)

        tp += len(pred_letters & gold)
        fp += len(pred_letters - gold) + len(unmapped)
        fn += len(gold - pred_letters)

        results.append({
            "question": row["question"],
            "answers": row["answers"],
            "ground_truth": sorted(gold),
            "pred_intents": names,
            "pred_letters": sorted(pred_letters),
            "unmapped_intents": unmapped,
            "parser_tier": tier,
            "correct": pred_letters == gold and not unmapped,
            "raw_output": raw,
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(rows)

    summary = {
        "task": "2.5-native-prompt",
        "nudge": bool(args.nudge),
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision_mode": "bf16",
        "backend": "vllm",
        "data": str(args.data),
        "num_examples": n,
        "exact_match": exact / n,
        "micro": {"precision": precision, "recall": recall, "micro_f1": micro_f1, "tp": tp, "fp": fp, "fn": fn},
        "invalid": invalid,
        "unmapped_intent_count": unmapped_total,
        "rows_with_unmapped": rows_with_unmapped,
        "parser_tiers": tier_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Task 2.5 — prompt native (JSON tên intent, như lúc train) ===")
    print(f"Exact Match : {exact}/{n} = {exact/n:.2%}")
    print(f"Micro-F1    : {micro_f1:.1%}  (P={precision:.1%} R={recall:.1%}  TP={tp} FP={fp} FN={fn})")
    print(f"Invalid (không parse được): {invalid}/{n}")
    print(f"Sinh intent ngoài 4 lựa chọn: {unmapped_total} nhãn, ở {rows_with_unmapped}/{n} câu")
    print(f"Parser tiers: {tier_count}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
