#!/usr/bin/env python3
"""Validate Legal SFT JSONL without modifying it."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant"}


def to_messages(row: dict[str, Any], default_system: str = "") -> list[dict[str, str]]:
    if "messages" in row:
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages phải là list không rỗng")
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("mỗi message phải là object")
            role, content = message.get("role"), message.get("content")
            if role not in ALLOWED_ROLES or not isinstance(content, str) or not content.strip():
                raise ValueError(f"message không hợp lệ: role={role!r}")
            normalized.append({"role": role, "content": content.strip()})
        if normalized[-1]["role"] != "assistant":
            raise ValueError("message cuối phải có role=assistant")
        return normalized

    if "instruction" in row and "output" in row:
        instruction = row["instruction"]
        output = row["output"]
        input_text = row.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction phải là chuỗi không rỗng")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("output phải là chuỗi không rỗng")
        user = instruction.strip()
        if isinstance(input_text, str) and input_text.strip():
            user += "\n\n" + input_text.strip()
        messages = []
        if default_system:
            messages.append({"role": "system", "content": default_system})
        messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": output.strip()},
            ]
        )
        return messages
    raise ValueError("cần có messages hoặc instruction/output")


def load_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dòng {line_number}: JSON phải là object")
            rows.append((line_number, value))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--model", help="Model ID/path để thống kê token")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--default-system", default="")
    args = parser.parse_args()
    tokenizer = None
    if args.model:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Thiếu transformers. Hãy cài requirements.txt trước khi thống kê token."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    failed = False

    for path in args.files:
        lengths: list[int] = []
        errors: list[str] = []
        try:
            rows = load_rows(path)
        except Exception as exc:
            print(f"ERROR {path}: {exc}")
            failed = True
            continue
        for line_number, row in rows:
            try:
                messages = to_messages(row, args.default_system)
                if tokenizer:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    lengths.append(len(tokenizer(text, add_special_tokens=False).input_ids))
            except Exception as exc:
                errors.append(f"dòng {line_number}: {exc}")
        print(f"{path}: rows={len(rows)}, invalid={len(errors)}")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... còn {len(errors) - 20} lỗi")
        if lengths:
            over = sum(length > args.max_length for length in lengths)
            print(
                f"  tokens min/median/max={min(lengths)}/"
                f"{statistics.median(lengths):.0f}/{max(lengths)}; "
                f">{args.max_length}={over} ({over / len(lengths):.1%})"
            )
        failed |= bool(errors) or not rows
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
