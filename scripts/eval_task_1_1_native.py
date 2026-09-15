#!/usr/bin/env python3
"""Chấm benchmark task 1.1 (trích xuất thực thể) bằng PROMPT NATIVE lúc train,
thay vì ép model chọn trắc nghiệm A-D.

Cách làm:
  1. System prompt lấy nguyên văn từ nhóm V2 trong file train (không hard-code).
  2. Bóc đoạn văn ra khỏi lớp vỏ câu hỏi ("Hãy trích xuất ... dưới đây: \"...\"")
     để đưa vào đúng format lúc train (user = đoạn văn trần).
  3. Model sinh JSON {"entities":[{"text","type"}]} tự do.
  4. Map ngược: so tập thực thể model sinh với 4 tập ứng viên A-D, chọn tập khớp
     nhất (Micro-F1 cao nhất) làm đáp án, rồi so với ground_truth.

Lưu ý thiết kế: 4 lựa chọn chỉ liệt kê một tập con thực thể của đoạn văn (phần
"tranh chấp" giữa các phương án), trong khi model sinh toàn bộ thực thể nó thấy.
Nên khi so khớp, tập model sinh được LỌC chỉ giữ các thực thể có mặt trong ít
nhất một lựa chọn — để không phạt oan model vì trích thêm thực thể đúng nhưng
nằm ngoài phạm vi 4 phương án.
"""

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ENTITY_TYPES = {
    "PERSON", "LEGAL_ROLE", "STATE_BODY", "ORGANIZATION", "COURT", "LAW",
    "LEGAL_DOCUMENT", "LEGAL_PROVISION", "CASE_ID", "DATE", "MONEY", "LOCATION",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prompt-source", type=Path, default=Path("data-finetune-v2/data_finetune_v2_train.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--split-hint",
        action="store_true",
        help="Thêm 1 câu nhắc ngắn vào cuối system prompt: nếu nhiều thực thể cùng loại "
        "bị liệt kê liền nhau (cách nhau bởi dấu phẩy/chấm phẩy/và), phải tách thành các "
        "entity riêng biệt, không gộp chung — thử nghiệm không cần train lại, xem prompt "
        "có sửa được lỗi 'gộp cụm' hay không.",
    )
    return p.parse_args()


SPLIT_HINT = (
    "\n\nLưu ý quan trọng: nếu nhiều thực thể CÙNG LOẠI được liệt kê liền nhau trong "
    "cùng một câu/mệnh đề (cách nhau bởi dấu phẩy, dấu chấm phẩy, hoặc từ \"và\"), hãy "
    "tách MỖI thực thể thành một mục JSON riêng biệt — không gộp chung nhiều thực thể "
    "vào một chuỗi text."
)


def load_v2_system_prompt(path: Path, split_hint: bool = False) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("group") == "V2":
                prompt = row["messages"][0]["content"]
                return prompt + SPLIT_HINT if split_hint else prompt
    raise SystemExit("Không tìm thấy mẫu V2 nào trong file prompt-source.")


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).strip().split()).lower()


def extract_paragraph(question: str) -> str:
    """'Hãy trích xuất ... dưới đây: "<đoạn văn>"' -> '<đoạn văn>'"""
    first, last = question.find('"'), question.rfind('"')
    if first != -1 and last > first:
        return question[first + 1:last].strip()
    return question.split(":", 1)[-1].strip()


def parse_options(answers: str) -> dict[str, set[tuple[str, str]]]:
    """'A: TYPE: text; TYPE: text\\nB: ...' -> {'A': {(type, text), ...}, ...}"""
    options: dict[str, set[tuple[str, str]]] = {}
    for line in answers.split("\n"):
        m = re.match(r"\s*([A-D])\s*:\s*(.*)$", line)
        if not m:
            continue
        letter, body = m.group(1), m.group(2)
        items = set()
        for chunk in body.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            etype, text = chunk.split(":", 1)
            etype = etype.strip().upper()
            if etype in ENTITY_TYPES:
                items.add((etype, norm(text)))
        options[letter] = items
    return options


def parse_entities(raw: str) -> tuple[set[tuple[str, str]], str]:
    """-> (tập (type, text), tier parser)"""
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            ents = obj.get("entities")
            if isinstance(ents, list):
                out = {
                    (str(e.get("type", "")).strip().upper(), norm(e.get("text", "")))
                    for e in ents
                    if isinstance(e, dict) and str(e.get("type", "")).strip().upper() in ENTITY_TYPES
                }
                if out:
                    return out, "json"
        except json.JSONDecodeError:
            pass

    # fallback: quét cặp "TYPE: text" trong văn bản thô
    pairs = re.findall(r"\b(" + "|".join(ENTITY_TYPES) + r")\b\s*[:\-]\s*([^;\n\"}]+)", text)
    out = {(t.upper(), norm(v)) for t, v in pairs}
    if out:
        return out, "regex"
    return set(), "invalid"


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def main() -> None:
    args = parse_args()

    system_prompt = load_v2_system_prompt(args.prompt_source, split_hint=args.split_hint)
    rows = [json.loads(l) for l in args.data.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row in rows:
        paragraph = extract_paragraph(row["question"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": paragraph}],
                tokenize=False, add_generation_prompt=True,
            )
        )

    llm_kwargs: dict[str, Any] = {}
    lora_request = None
    if args.adapter is not None:
        cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        llm_kwargs.update(enable_lora=True, max_lora_rank=int(cfg.get("r", 16)), max_loras=1)
        lora_request = LoRARequest("legal-adapter", 1, str(args.adapter))

    llm = LLM(
        model=str(args.model), dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **llm_kwargs,
    )
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
                           lora_request=lora_request)

    correct = 0
    invalid = 0
    ties = 0
    total_generated = 0
    tier_count: dict[str, int] = {}
    results = []
    tp = fp = fn = 0

    for row, out in zip(rows, outputs):
        raw = out.outputs[0].text.strip()
        pred_ents, tier = parse_entities(raw)
        tier_count[tier] = tier_count.get(tier, 0) + 1
        invalid += int(tier == "invalid")
        total_generated += len(pred_ents)

        options = parse_options(row["answers"])
        universe = set().union(*options.values()) if options else set()
        restricted = pred_ents & universe   # chỉ so trên vùng tranh chấp giữa 4 phương án

        scores = {letter: f1(restricted, opt) for letter, opt in options.items()}
        best = max(scores.values()) if scores else 0.0
        winners = [l for l, s in scores.items() if s == best]
        ties += int(len(winners) > 1)
        prediction = sorted(winners)[0] if winners else None

        truth = str(row["ground_truth"]).strip().upper()
        is_correct = prediction == truth
        correct += int(is_correct)

        # Micro-F1: so trực tiếp tập thực thể model sinh (đã lọc vào vùng tranh
        # chấp) với tập thực thể của ĐÁP ÁN ĐÚNG — cho điểm từng phần thay vì
        # chỉ đúng/sai theo chữ cái đã chọn.
        gold_ents = options.get(truth, set())
        tp += len(restricted & gold_ents)
        fp += len(restricted - gold_ents)
        fn += len(gold_ents - restricted)

        results.append({
            "question": row["question"][:200],
            "ground_truth": truth,
            "prediction": prediction,
            "correct": is_correct,
            "option_scores": scores,
            "n_entities_generated": len(pred_ents),
            "n_entities_in_universe": len(restricted),
            "tie": len(winners) > 1,
            "parser_tier": tier,
            "raw_output": raw[:500],
        })

    n = len(rows)
    micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) else 0.0
    )
    summary = {
        "task": "1.1-native-prompt",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision_mode": "bf16",
        "backend": "vllm",
        "data": str(args.data),
        "num_examples": n,
        "correct": correct,
        "accuracy": correct / n,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "invalid": invalid,
        "ties": ties,
        "avg_entities_generated": total_generated / n,
        "parser_tiers": tier_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Task 1.1 — prompt native (JSON entities, như lúc train) ===")
    print(f"Micro-F1 : {micro_f1:.2%} (P={micro_precision:.2%}, R={micro_recall:.2%})")
    print(f"Accuracy (chọn đúng chữ cái): {correct}/{n} = {correct/n:.2%}")
    print(f"Invalid (không parse được): {invalid}/{n}")
    print(f"Hoà điểm giữa các phương án (tie): {ties}/{n}")
    print(f"Trung bình số thực thể model sinh: {total_generated/n:.2f}")
    print(f"Parser tiers: {tier_count}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
