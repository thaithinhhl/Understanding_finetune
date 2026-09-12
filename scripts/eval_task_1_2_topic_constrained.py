#!/usr/bin/env python3
"""Chấm task 1.2 bằng cách ép model chỉ được chọn 1 trong 6 lĩnh vực đã cho (không tự
bịa lĩnh vực khác), nhưng vẫn trả lời THEO ĐÚNG FORMAT JSON đã học lúc train.

Cách làm: với mỗi câu hỏi, chấm điểm (log-likelihood) cho cả 6 ứng viên
`{"topic":"<tên lĩnh vực đã cho>"}` bằng 1 forward pass (teacher-forcing), rồi chọn
ứng viên có log-likelihood cao nhất — tương đương "constrained decoding" ở mức
chuỗi JSON đầy đủ thay vì ở mức 1 token chữ cái. Không còn khả năng bịa lĩnh vực
ngoài 6 lựa chọn (khác với scripts/eval_task_1_2_topic_mapping.py).
"""

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "Bạn phân loại lĩnh vực pháp luật của câu hỏi tiếng Việt. Chỉ trả về JSON:\n\n"
    '{"topic": string}\n\n'
    "`topic` phải là một trong các lĩnh vực sau:\n"
    "- Bảo hiểm\n- Bất động sản\n- Bộ máy hành chính\n- Chứng khoán\n"
    "- Công nghệ thông tin\n- Doanh nghiệp\n- Dịch vụ pháp lý\n- Giao thông vận tải\n"
    "- Giáo dục\n- Kế toán - kiểm toán\n- Lao động - tiền lương\n- Lĩnh vực khác\n"
    "- Quyền dân sự\n- Sở hữu trí tuệ\n- Thuế - phí - lệ phí\n- Thương mại\n"
    "- Thể thao - y tế\n- Thủ tục tố tụng\n- Tiền tệ ngân hàng\n- Trách nhiệm hình sự\n"
    "- Tài chính nhà nước\n- Tài nguyên - môi trường\n- Vi phạm hành chính\n"
    "- Văn hóa - xã hội\n- Xuất nhập khẩu\n- Xây dựng - đô thị\n- Đầu tư\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--precision", choices=("bf16", "4bit"), default="4bit")
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.strip().split())


def parse_choices(answers: str) -> dict[str, str]:
    parts = re.findall(r"([A-F])\.\s*(.+?)(?=\s+[A-F]\.\s|$)", answers.strip())
    return {letter: norm(text) for letter, text in parts}


def score_candidates(
    model, tokenizer, prompt_ids: list[int], candidate_texts: list[str]
) -> list[float]:
    """Trả về tổng log-likelihood (teacher-forcing) của từng candidate, batch chung 1 forward pass."""
    device = model.device
    target_ids_list = [
        tokenizer(text, add_special_tokens=False).input_ids for text in candidate_texts
    ]
    pad_id = tokenizer.pad_token_id
    full_seqs = [prompt_ids + t for t in target_ids_list]
    max_len = max(len(s) for s in full_seqs)

    input_ids = torch.full((len(full_seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(full_seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(full_seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    scores = []
    prompt_len = len(prompt_ids)
    for i, target_ids in enumerate(target_ids_list):
        n = len(target_ids)
        # logits[prompt_len-1 : prompt_len-1+n] dự đoán target_ids[0:n]
        step_logits = logits[i, prompt_len - 1 : prompt_len - 1 + n, :]
        log_probs = F.log_softmax(step_logits.float(), dim=-1)
        target = torch.tensor(target_ids, device=device)
        token_logp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        scores.append(token_logp.sum().item())
    return scores


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch không nhìn thấy CUDA GPU.")

    rows = load_rows(args.data)
    selected = rows[args.offset:] if args.limit is None else rows[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("Không có mẫu nào trong khoảng offset/limit đã chọn.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    correct = 0
    for position, row in enumerate(selected, args.offset):
        choices = parse_choices(row["answers"])  # {'A': 'dịch vụ pháp lý', ...} (đã norm/lower)
        letters = list(choices.keys())
        # dùng đúng chữ hoa/thường gốc trong answers để build JSON target tự nhiên hơn
        raw_choices = re.findall(r"[A-F]\.\s*(.+?)(?=\s+[A-F]\.\s|$)", row["answers"].strip())
        candidate_texts = [f'{{"topic":"{name.strip()}"}}' for name in raw_choices]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        scores = score_candidates(model, tokenizer, prompt_ids, candidate_texts)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        prediction = letters[best_idx]
        truth = str(row["ground_truth"]).strip().upper()
        is_correct = prediction == truth
        correct += int(is_correct)
        results.append({
            "index": position, "question": row["question"], "answers": row["answers"],
            "ground_truth": truth, "prediction": prediction, "scores": scores,
            "correct": is_correct,
        })
        print(
            f"[{len(results):03d}/{len(selected):03d}] pred={prediction} "
            f"truth={truth} correct={is_correct}"
        )

    accuracy = correct / len(results)
    summary = {
        "task": "1.2-topic-constrained", "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": args.precision, "data": str(args.data),
        "num_examples": len(results), "correct": correct,
        "accuracy": accuracy, "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.2%}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
