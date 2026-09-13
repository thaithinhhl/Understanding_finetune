#!/usr/bin/env python3
"""Chấm task 1.2 giữ NGUYÊN system prompt lúc train (V1: JSON {"topic": ...}, 26
lĩnh vực), chỉ THÊM vào user message danh sách 6 lựa chọn cụ thể của câu hỏi này
-- khác với scripts/eval_task_1_2_topic_mapping.py (không hề cho model biết 6 lựa
chọn) và scripts/eval_task_1_2_topic_constrained.py (chỉ dùng 6 lựa chọn lúc CHẤM
ĐIỂM log-likelihood, không đưa vào prompt cho model thấy lúc sinh).

Ý tưởng: không ép format lạ (constrained decoding 1 chữ cái, model chưa từng học),
không bỏ qua thông tin đề bài đã cho (6 lựa chọn) -- để model tự phân loại theo
đúng phản xạ đã train, nhưng có đủ ngữ cảnh để chọn đúng trong số 6 lựa chọn thay
vì lạc sang 1 trong 26 lĩnh vực không có trong đề.
"""

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
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--precision", choices=("bf16", "4bit"), default="bf16")
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
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch không nhìn thấy CUDA GPU.")

    rows = load_rows(args.data)
    selected = rows[args.offset:] if args.limit is None else rows[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("Không có mẫu nào trong khoảng offset/limit đã chọn.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
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
    for position, row in enumerate(selected, args.offset):
        choices = parse_choices(row["answers"])
        raw_choices = re.findall(r"[A-F]\.\s*(.+?)(?=\s+[A-F]\.\s|$)", row["answers"].strip())
        choice_list = "\n".join(f"- {name.strip()}" for name in raw_choices)
        user_content = (
            f"{row['question']}\n\n"
            f"Lưu ý: câu hỏi này chỉ thuộc một trong các lĩnh vực sau, hãy chọn đúng lĩnh vực "
            f"từ danh sách này (giữ nguyên cách viết):\n{choice_list}\n\n"
            f"BẮT BUỘC: giá trị `topic` phải là NGUYÊN VĂN một trong các lĩnh vực liệt kê ở trên. "
            f"TUYỆT ĐỐI KHÔNG được chọn hoặc bịa ra lĩnh vực nào khác ngoài danh sách này."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            "index": position, "question": row["question"], "answers": row["answers"],
            "ground_truth": truth, "predicted_topic": topic, "prediction": prediction,
            "correct": is_correct, "raw_output": raw_output,
        })
        print(
            f"[{len(results):03d}/{len(selected):03d}] topic={topic!r} "
            f"pred={prediction or 'UNMAPPED'} truth={truth} correct={is_correct}"
        )

    accuracy = correct / len(results)
    summary = {
        "task": "1.2-native-informed", "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": args.precision, "data": str(args.data),
        "num_examples": len(results), "correct": correct, "unmapped": unmapped,
        "accuracy": accuracy, "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.2%}")
    print(f"Unmapped (topic không khớp lựa chọn nào): {unmapped}/{len(results)}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
