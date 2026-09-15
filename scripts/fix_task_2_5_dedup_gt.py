#!/usr/bin/env python3
"""Sửa lỗi ground-truth mâu thuẫn giữa các câu hỏi gần-trùng-lặp trong benchmark
task 2.5 (áp dụng SAU khi đã sửa lỗi chitchat, xem fix_task_2_5_chitchat_gt.py).

BỆNH ĐÃ CHẨN ĐOÁN (2026-09-15): nhiều câu hỏi gốc bị nhân bản thành nhiều dòng
benchmark (mỗi dòng chỉ đưa 4/8 intent làm lựa chọn), nhưng ground_truth giữa
các bản sao MÂU THUẪN NHAU THẬT SỰ — không chỉ vì offer intent khác nhau, mà
cùng 1 intent được offer ở 2 dòng lại bị đánh dấu đúng/sai khác nhau. Ví dụ đã
xác nhận: 3 dòng chứa nguyên văn 1 câu hỏi thừa kế, "comparative_analysis" vừa
là đáp án đúng (dòng 12, 114) vừa bị đánh sai (dòng 553) dù cùng có mặt trong
lựa chọn của cả 2 nơi.

CÁCH SỬA: gom các dòng gần-trùng-lặp (Jaccard >= 0.90, lấy từ
results/task-2.5-near-duplicates-jaccard-090.jsonl) thành cụm connected-
components (bắc cầu A~B~C), dùng Claude Sonnet xác định "tập intent ĐÚNG THẬT"
cho câu hỏi đại diện của mỗi cụm (không giới hạn theo 4 lựa chọn của bất kỳ
dòng nào), rồi áp lại cho từng dòng: ground_truth_moi = tap_dung_that ∩
4_lua_chon_cua_dong_do. Chỉ những dòng nằm trong 1 cụm >=2 phần tử mới bị đổi;
mọi dòng khác giữ nguyên.
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

ALL_INTENTS = [
    "chitchat", "comparative_analysis", "document_relationship", "document_retrieval",
    "external_analysis", "general", "legal_query", "stats_summary",
]
INTENT_DESC = {
    "chitchat": "Câu hỏi không liên quan đến pháp luật (chào hỏi, cảm ơn, off-topic)",
    "comparative_analysis": "So sánh nội dung giữa hai văn bản, điều khoản, nội dung, ...",
    "document_relationship": "Mối quan hệ giữa các văn bản: sửa đổi, bổ sung, hướng dẫn, dẫn chiếu, căn cứ",
    "document_retrieval": "Truy xuất toàn văn bản pháp luật",
    "external_analysis": "Tác động kinh tế, xã hội, xu hướng thay đổi, lịch sử, ảnh hưởng",
    "general": "Câu hỏi tổng quát liên quan pháp luật, chưa thuộc intent nào cụ thể",
    "legal_query": "Tìm và trả lời từ nội dung cụ thể của điều/khoản/mục/điểm cụ thể",
    "stats_summary": "Thống kê số lượng văn bản/quy định",
}

CLASSIFY_SYSTEM = (
    "Bạn xác định TẤT CẢ intent thực sự áp dụng cho 1 câu hỏi tiếng Việt, trong "
    "số 8 intent sau (một câu có thể có nhiều intent đúng cùng lúc):\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENT_DESC.items())
    + "\n\nChỉ trả về JSON: {\"intents\": [\"...\"], \"reason\": \"1 câu ngắn giải thích\"}. "
    "Chỉ liệt kê intent nào THỰC SỰ áp dụng, không đoán thêm."
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


def build_clusters(near_dup_path: Path) -> list[set[int]]:
    """Union-Find tren cac cap (source_line, similar.source_line) -> danh sach
    cum (moi cum la 1 set cac line index, 1-indexed)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    with near_dup_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            a = d["source_line"]
            for sim in d["similar_cases"]:
                union(a, sim["source_line"])

    groups: dict[int, set[int]] = {}
    for x in parent:
        groups.setdefault(find(x), set()).add(x)
    return [g for g in groups.values() if len(g) >= 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data-benchmark-v2/vlegal_task_2_5_test_chitchat_fixed.jsonl"))
    ap.add_argument("--near-dup", type=Path, default=Path("results/task-2.5-near-duplicates-jaccard-090.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data-benchmark-v2/vlegal_task_2_5_test_v2_gtfixed.jsonl"))
    ap.add_argument("--audit-log", type=Path, default=Path("data-benchmark-v2/vlegal_task_2_5_dedup_fix_audit.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    clusters = build_clusters(args.near_dup)
    print(f"So cum (>=2 dong): {len(clusters)}, tong so dong lien quan: {sum(len(c) for c in clusters)}")

    lock = threading.Lock()
    cluster_intents: dict[int, tuple[list[str], str]] = {}  # cluster_id -> (intents, reason)
    stats = {"api_err": 0, "parse_err": 0}

    def classify_cluster(cluster_id: int, lines: set[int]) -> None:
        # lay cau hoi dai dien = cau dai nhat trong cum (thuong day du nhat)
        rep_idx = max(lines, key=lambda ln: len(rows[ln - 1]["question"]))
        question = rows[rep_idx - 1]["question"]
        try:
            resp = client.messages.create(
                model=args.model, max_tokens=300,
                system=CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": question}],
            )
        except Exception as e:
            with lock:
                stats["api_err"] += 1
            cluster_intents[cluster_id] = (None, f"api_error: {e}")
            return
        text = "".join(b.text for b in resp.content if b.type == "text")
        obj = extract_json(text)
        if obj is None or not isinstance(obj.get("intents"), list):
            with lock:
                stats["parse_err"] += 1
            cluster_intents[cluster_id] = (None, "parse_error")
            return
        intents = [str(i).strip() for i in obj["intents"] if str(i).strip() in ALL_INTENTS]
        reason = str(obj.get("reason", ""))[:200]
        cluster_intents[cluster_id] = (intents, reason)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify_cluster, i, c): i for i, c in enumerate(clusters)}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 20 == 0:
                print(f"  ... {i}/{len(clusters)} cum", flush=True)

    audit = []
    n_changed = 0
    for cluster_id, lines in enumerate(clusters):
        true_intents, reason = cluster_intents[cluster_id]
        if true_intents is None:
            continue  # loi API/parse -> giu nguyen cum nay, khong sua
        for ln in sorted(lines):
            row = rows[ln - 1]
            letter_to_intent = parse_letter_to_intent(row["answers"])
            offered_letters = sorted(letter_to_intent)
            new_letters = sorted(l for l in offered_letters if letter_to_intent[l] in true_intents)
            old_letters = sorted(str(x).strip().upper() for x in row["ground_truth"])
            if new_letters != old_letters:
                n_changed += 1
            audit.append({
                "cluster_id": cluster_id,
                "source_line": ln,
                "question": row["question"],
                "answers": row["answers"],
                "old_ground_truth": old_letters,
                "true_intents_for_cluster": true_intents,
                "reason": reason,
                "new_ground_truth": new_letters,
            })
            row["ground_truth"] = new_letters if new_letters else old_letters
            if not new_letters:
                audit[-1]["note"] = "KHONG co lua chon nao khop tap intent dung -> giu GT cu de tranh rong"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.audit_log.open("w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"\nSo dong GT thay doi: {n_changed}/{sum(len(c) for c in clusters)}")
    print(f"Loi: api={stats['api_err']} parse={stats['parse_err']}")
    print(f"Da ghi {len(rows)} dong -> {args.output}")
    print(f"Audit log -> {args.audit_log}")


if __name__ == "__main__":
    main()
