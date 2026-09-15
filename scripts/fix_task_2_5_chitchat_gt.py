#!/usr/bin/env python3
"""Sửa lỗi ground-truth "chitchat" tự mâu thuẫn trong benchmark task 2.5
(vlegal_task_2_5_test.jsonl).

BỆNH ĐÃ CHẨN ĐOÁN (2026-09-15): 166/1359 dòng có "chitchat" trong ground_truth,
nhưng 0/166 dòng có chitchat đứng một mình — luôn đi kèm >=1 intent khác. Về
định nghĩa, chitchat = "câu hỏi không liên quan đến pháp luật", nên về logic
phải loại trừ lẫn nhau với mọi intent pháp lý khác. Kiểm tra thủ công một phần
cho thấy KHÔNG có quy tắc máy móc chung đúng cho cả 166 dòng:
  - Nhiều dòng nội dung THẬT SỰ không liên quan pháp luật (chào hỏi, công thức
    nấu ăn, lịch tập gym, thuật ngữ kỹ thuật như "websocket"...) -> chitchat mới
    đúng, các intent pháp lý đi kèm mới là nhãn sai.
  - Nhiều dòng khác nội dung THẬT SỰ là câu hỏi pháp lý (dù viết tắt/thô tục/
    ngắn) -> chitchat mới là nhãn sai, các intent còn lại mới đúng.

Script này dùng Claude Sonnet để phân loại từng dòng theo NỘI DUNG (không dựa
theo nhãn hiện có, tránh anchoring bias), rồi áp đúng 1 trong 2 quy tắc trên.
Ghi ra file mới, không đè lên benchmark gốc, kèm log lý do từng dòng để review.
"""

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

CLASSIFY_SYSTEM = (
    "Bạn phân loại xem một câu hỏi/văn bản tiếng Việt có THỰC SỰ liên quan đến "
    "pháp luật Việt Nam hay không (bất kể văn bản viết tắt, thô tục, ngắn gọn, "
    "sai chính tả). Chỉ trả về đúng JSON:\n"
    '{"is_legal": true/false, "reason": "1 câu ngắn giải thích"}\n\n'
    "is_legal=true nếu nội dung hỏi/đề cập đến luật, nghị định, quyền/nghĩa vụ, "
    "thủ tục hành chính/tố tụng, hợp đồng, tài chính-thuế có yếu tố pháp lý, tranh "
    "chấp, hình phạt, hoặc bất kỳ chủ đề pháp lý nào — kể cả khi câu hỏi viết rất "
    "tắt/thô/ngắn (ví dụ 'văn bản căn cứ j', 'ly hôn mất bao lâu').\n"
    "is_legal=false nếu nội dung là chào hỏi xã giao, cảm ơn, trò chuyện phiếm, "
    "hỏi về nấu ăn/thể thao/công nghệ chung/kiến thức phổ thông không dính pháp "
    "luật, hoặc câu vô nghĩa/lệnh cho hệ thống không liên quan pháp luật (ví dụ "
    "'Hi', 'Dài hơn', 'Công thức sườn xào chua ngọt', 'Web socket là gì')."
)


def parse_letter_to_intent(answers: str) -> dict[str, str]:
    return dict(re.findall(r"([A-D])\.\s*(\S+)", answers.strip()))


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("vlegal_task_2_5_test.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data-benchmark-v2/vlegal_task_2_5_test_chitchat_fixed.jsonl"))
    ap.add_argument("--audit-log", type=Path, default=Path("data-benchmark-v2/vlegal_task_2_5_chitchat_fix_audit.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()

    rows = []
    with args.data.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    targets = []  # (line_idx0, gold_letters, gold_intents_without_chitchat, has_chitchat_and_other)
    for idx, row in enumerate(rows):
        letter_to_intent = parse_letter_to_intent(row["answers"])
        gold_letters = [str(l).strip().upper() for l in row["ground_truth"]]
        gold_intents = [letter_to_intent.get(l, "?") for l in gold_letters]
        if "chitchat" in gold_intents and len(gold_intents) > 1:
            targets.append((idx, letter_to_intent, gold_letters, gold_intents))

    print(f"Tong so dong can sua: {len(targets)}")

    lock = threading.Lock()
    audit: list[dict] = []
    stats = {"api_err": 0, "parse_err": 0, "legal": 0, "nonlegal": 0}

    def classify_one(item):
        idx, letter_to_intent, gold_letters, gold_intents = item
        row = rows[idx]
        question = row["question"]
        try:
            resp = client.messages.create(
                model=args.model, max_tokens=200,
                system=CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": question}],
            )
        except Exception as e:
            with lock:
                stats["api_err"] += 1
            return idx, None, f"api_error: {e}"

        text = "".join(b.text for b in resp.content if b.type == "text")
        obj = extract_json(text)
        if obj is None or "is_legal" not in obj:
            with lock:
                stats["parse_err"] += 1
            return idx, None, "parse_error"

        is_legal = bool(obj["is_legal"])
        reason = str(obj.get("reason", ""))[:200]
        with lock:
            stats["legal" if is_legal else "nonlegal"] += 1
        return idx, is_legal, reason

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify_one, item): item for item in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            idx, is_legal, reason = fut.result()
            results[idx] = (is_legal, reason)
            if i % 30 == 0:
                print(f"  ... {i}/{len(targets)}", flush=True)

    # ap dung fix
    letter_map_cache = {}
    for idx, letter_to_intent, gold_letters, gold_intents in targets:
        is_legal, reason = results[idx]
        row = rows[idx]
        intent_to_letter = {v: k for k, v in letter_to_intent.items()}
        if is_legal is None:
            # khong phan loai duoc -> giu nguyen, khong sua, ghi log de xu ly thu cong
            new_intents = gold_intents
            action = "SKIPPED (loi API/parse)"
        elif is_legal:
            new_intents = [i for i in gold_intents if i != "chitchat"]
            action = "DROP chitchat (noi dung la phap ly)"
        else:
            new_intents = ["chitchat"]
            action = "KEEP only chitchat (noi dung khong lien quan phap luat)"
        new_letters = sorted({intent_to_letter[i] for i in new_intents if i in intent_to_letter})
        old_letters = sorted(gold_letters)

        audit.append({
            "source_line": idx + 1,
            "question": row["question"],
            "answers": row["answers"],
            "old_ground_truth": old_letters,
            "old_intents": gold_intents,
            "is_legal_content": is_legal,
            "reason": reason,
            "action": action,
            "new_ground_truth": new_letters,
            "new_intents": new_intents,
        })
        row["ground_truth"] = new_letters

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.audit_log.open("w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"\nPhan loai: {stats['legal']} la cau hoi phap ly (bo chitchat), "
          f"{stats['nonlegal']} khong lien quan phap luat (chi giu chitchat), "
          f"loi: api={stats['api_err']} parse={stats['parse_err']}")
    print(f"Da ghi {len(rows)} dong -> {args.output}")
    print(f"Audit log (166 dong da sua, kem ly do) -> {args.audit_log}")


if __name__ == "__main__":
    main()
