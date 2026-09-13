#!/usr/bin/env python3
"""Chấm benchmark task 1.2 (trắc nghiệm A-F lĩnh vực pháp luật) bằng vLLM, dùng
đúng prompt chung lưu trong prompt_gold_task1.2.json (tách System:/User:, thay
{question}/{choice_list}). Model sinh JSON {"topic": "..."} tự do, sau đó map
ngược về chữ cái khớp với 6 lựa chọn. Hỗ trợ nạp thêm LoRA adapter và model đã
lượng tử hoá sẵn (AWQ)."""

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


ANSWER_LETTERS = "ABCDEF"

TRAIN_SYSTEM_TEMPLATE = (
    'Bạn phân loại lĩnh vực pháp luật của câu hỏi tiếng Việt. Chỉ trả về JSON:\n\n'
    '{"topic": string}\n\n'
    "`topic` phải là một trong các lĩnh vực sau:\n{topic_list}"
)

# Đủ 26 lĩnh vực, lấy nguyên văn từ system prompt lúc train nhóm V1
# (data-finetune-v2/data_finetune_v2_train.jsonl) — dùng cho --train-prompt mặc định.
ALL_26_TOPICS = [
    "Bảo hiểm", "Bất động sản", "Bộ máy hành chính", "Chứng khoán",
    "Công nghệ thông tin", "Doanh nghiệp", "Dịch vụ pháp lý", "Giao thông vận tải",
    "Giáo dục", "Kế toán - kiểm toán", "Lao động - tiền lương", "Lĩnh vực khác",
    "Quyền dân sự", "Sở hữu trí tuệ", "Thuế - phí - lệ phí", "Thương mại",
    "Thể thao - y tế", "Thủ tục tố tụng", "Tiền tệ ngân hàng", "Trách nhiệm hình sự",
    "Tài chính nhà nước", "Tài nguyên - môi trường", "Vi phạm hành chính",
    "Văn hóa - xã hội", "Xuất nhập khẩu", "Xây dựng - đô thị", "Đầu tư",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter tùy chọn để nạp lên trên --model.")
    p.add_argument("--prompt-file", type=Path, default=Path("prompt_gold_task1.2.json"))
    p.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Bỏ qua --prompt-file, dùng nguyên instruction+question+answers từ benchmark "
        "làm user message (không system prompt), chấm bằng cách trích chữ cái trực tiếp.",
    )
    p.add_argument(
        "--train-prompt",
        action="store_true",
        help="Dùng đúng system prompt lúc train nhóm V1 (data-finetune-v2): mặc định liệt kê "
        "đủ 26 lĩnh vực như lúc train thật, model sinh JSON {\"topic\":...} tự do rồi map "
        "ngược sang chữ cái khớp với 1 trong 6 lựa chọn của câu hỏi (nếu topic sinh ra không "
        "khớp lựa chọn nào thì tính là sai/unmapped). User message = câu hỏi trần, giống hệt "
        "format lúc train. Thêm --narrow-topics để thu hẹp còn đúng 6 lựa chọn (biến thể so sánh).",
    )
    p.add_argument(
        "--narrow-topics",
        action="store_true",
        help="Chỉ có tác dụng cùng --train-prompt: thu hẹp danh sách lĩnh vực trong system "
        "prompt xuống còn đúng 6 lựa chọn của câu hỏi benchmark thay vì đủ 26 lĩnh vực lúc train.",
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--label", default=None, help="Tên hiển thị trong summary (mặc định dùng --model).")
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


def extract_answer(text: str) -> str | None:
    normalized = text.strip().upper()
    match = re.search(r"(?:ĐÁP ÁN|ANSWER)\s*(?:LÀ|IS|:)?\s*([A-F])\b", normalized)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([A-F])(?:\s|[.):-]|$)", normalized)
    return match.group(1) if match else None


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


def main() -> None:
    args = parse_args()
    if args.raw_prompt and args.train_prompt:
        raise SystemExit("Chỉ được chọn một trong --raw-prompt hoặc --train-prompt.")

    mode = "raw" if args.raw_prompt else ("train" if args.train_prompt else "prompt_gold")

    system_prompt = user_template = None
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
    choices_per_row = []
    for row in selected:
        choices = parse_choices(row["answers"])
        choices_per_row.append(choices)
        raw_choices = re.findall(r"[A-F]\.\s*(.+?)(?=\s+[A-F]\.\s|$)", row["answers"].strip())

        if mode == "raw":
            user_content = "\n\n".join([row["instruction"], row["question"], row["answers"]])
            messages = [{"role": "user", "content": user_content}]
        elif mode == "train":
            listed_topics = [name.strip() for name in raw_choices] if args.narrow_topics else ALL_26_TOPICS
            topic_list = "\n".join(f"- {name}" for name in listed_topics)
            system_content = TRAIN_SYSTEM_TEMPLATE.replace("{topic_list}", topic_list)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": row["question"]},
            ]
        else:
            choice_list = "\n".join(f"- {name.strip()}" for name in raw_choices)
            user_content = user_template.replace("{question}", row["question"]).replace("{choice_list}", choice_list)
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
    unmapped = 0
    for row, choices, out in zip(selected, choices_per_row, outputs):
        raw_output = out.outputs[0].text.strip()
        if mode == "raw":
            topic = None
            prediction = extract_answer(raw_output)
        else:
            topic = extract_topic(raw_output)
            prediction = map_topic_to_letter(topic, choices)
        truth = str(row["ground_truth"]).strip().upper()
        is_correct = prediction == truth
        correct += int(is_correct)
        unmapped += int(prediction is None)
        results.append(
            {
                "question": row["question"][:200],
                "ground_truth": truth,
                "predicted_topic": topic,
                "prediction": prediction,
                "correct": is_correct,
                "raw_output": raw_output,
            }
        )

    n = len(results)
    accuracy = correct / n
    prompt_desc = {
        "raw": "raw (instruction+question+answers, no system prompt)",
        "train": (
            "train (V1 training system prompt, 6-choice topic list)"
            if args.narrow_topics
            else "train (V1 training system prompt, full 26-topic list as in actual training)"
        ),
        "prompt_gold": str(args.prompt_file),
    }[mode]
    summary = {
        "task": "1.2-prompt-gold-vllm",
        "model": args.label or str(args.model),
        "model_path": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": "awq" if is_prequantized else "bf16",
        "backend": "vllm",
        "prompt_mode": mode,
        "prompt_file": prompt_desc,
        "data": str(args.data),
        "num_examples": n,
        "correct": correct,
        "unmapped": unmapped,
        "accuracy": accuracy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {summary['model']} ===")
    print(f"Accuracy: {correct}/{n} = {accuracy:.2%}")
    print(f"Unmapped: {unmapped}/{n}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
