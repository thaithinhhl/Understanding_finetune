#!/usr/bin/env python3
"""Sinh lại dữ liệu train cho task 2.5 (nhóm V3 - user_intent_understanding).

Khắc phục 3 điểm yếu đã đo được ở bộ V3 hiện tại:

1. Phân bố số nhãn lệch: hiện 1.20 nhãn/câu (80% một nhãn, 0 mẫu 3-4 nhãn),
   trong khi thực tế cần nhiều nhãn hơn. Model học xong chỉ dám trả ~1.5 nhãn.
2. Chỉ phủ 6/28 tổ hợp nhãn khả dĩ, toàn tổ hợp 2 nhãn.
3. Thực thể lặp: 34 số hiệu văn bản dùng lại 362 lần, gán ngẫu nhiên cho chủ đề
   không liên quan.

Nguyên tắc gán nhãn giữ nguyên như bộ hiện tại (đúng ngữ nghĩa): mỗi nhãn ứng với
một mệnh đề thật trong câu hỏi, không gán thừa. chitchat/general không ghép với
intent khác vì chúng có nghĩa loại trừ.
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------- pools

MINISTRIES = [
    "TT-BYT", "TT-BTC", "TT-BQP", "TT-BTTTT", "TT-BGDĐT", "TT-BXD",
    "TT-BNV", "TT-BCA", "TT-BKHĐT", "TT-BTNMT", "TT-BLĐTBXH", "TT-BCT",
    "TT-BNNPTNT", "TT-BGTVT", "TT-BTP", "TT-NHNN",
]

TOPICS = [
    "chứng khoán và trái phiếu", "an toàn thực phẩm", "tuyển sinh đại học",
    "phòng cháy chữa cháy", "đấu thầu", "bảo vệ môi trường",
    "xuất nhập khẩu hàng hóa", "khám chữa bệnh bằng bảo hiểm y tế",
    "thừa kế tài sản", "sở hữu trí tuệ", "thuế thu nhập cá nhân",
    "đất đai và sổ đỏ", "đầu tư nước ngoài", "hôn nhân và gia đình",
    "bảo hiểm xã hội", "hợp đồng lao động", "xử phạt vi phạm giao thông",
    "đăng ký doanh nghiệp", "giấy phép xây dựng", "nghĩa vụ quân sự",
    "bảo vệ dữ liệu cá nhân", "phòng chống rửa tiền", "kinh doanh bất động sản",
    "an toàn lao động", "giáo dục nghề nghiệp", "quản lý thuế doanh nghiệp",
    "cấp phép lao động cho người nước ngoài", "bảo hiểm thất nghiệp",
    "tố tụng dân sự", "khiếu nại tố cáo", "chuyển đổi số",
    "quản lý tài sản công", "cạnh tranh không lành mạnh", "bảo vệ người tiêu dùng",
]

# ------------------------------------------------- mệnh đề theo từng intent
# Mỗi mệnh đề diễn đạt ĐÚNG MỘT intent. Dạng "mở" dùng khi đứng đầu câu,
# dạng "nối" dùng khi ghép sau mệnh đề khác.

CLAUSES = {
    "legal_query": {
        "open": [
            "cho tôi hỏi nội dung cụ thể của Điều {art} trong văn bản {doc} về {topic}",
            "cho tôi biết nội dung Điều {art} của văn bản {doc} quy định gì về {topic}",
            "Điều {art} trong văn bản {doc} nói gì về {topic}",
        ],
        "join": [
            "nội dung cụ thể của Điều {art} văn bản {doc} quy định thế nào",
            "Điều {art} của văn bản {doc} nêu cụ thể những gì",
        ],
    },
    "document_retrieval": {
        "open": [
            "cho tôi xin toàn văn văn bản {doc} về {topic}",
            "xin vui lòng cung cấp toàn văn văn bản {doc}",
            "cho tôi toàn văn văn bản {doc} với",
        ],
        "join": [
            "cho tôi xin toàn văn văn bản đó luôn",
            "gửi tôi toàn văn văn bản {doc}",
        ],
    },
    "document_relationship": {
        "open": [
            "cho mình hỏi Điều {art} của văn bản {doc} có văn bản nào sửa đổi hay hướng dẫn không",
            "Điều {art} văn bản {doc} có bị sửa đổi hay thay thế bởi văn bản nào không",
        ],
        "join": [
            "văn bản nào đã sửa đổi, thay thế hoặc hướng dẫn cho điều này",
            "có văn bản nào sửa đổi hay hướng dẫn thực hiện điều đó không",
        ],
    },
    "comparative_analysis": {
        "open": [
            "so sánh Điều {art} của văn bản {doc} với quy định trước đó về {topic}",
            "Điều {art} của văn bản {doc} về {topic} có gì khác so với quy định cũ",
        ],
        "join": [
            "so với quy định trước đó thì khác nhau ra sao",
            "điều này khác gì so với văn bản cũ về cùng nội dung",
        ],
    },
    "external_analysis": {
        "open": [
            "quy định về {topic} thay đổi đã tác động thế nào đến kinh tế - xã hội",
            "những thay đổi trong quy định về {topic} ảnh hưởng ra sao đến thị trường",
        ],
        "join": [
            "tác động của nó đến kinh tế - xã hội ra sao",
            "thay đổi này ảnh hưởng thế nào đến người dân và doanh nghiệp",
        ],
    },
    "stats_summary": {
        "open": [
            "xin hỏi hiện có bao nhiêu quy định về {topic}",
            "cho hỏi có bao nhiêu văn bản đang quy định về {topic}",
        ],
        "join": [
            "hiện có bao nhiêu quy định liên quan đến {topic}",
            "tổng cộng có bao nhiêu quy định về {topic} đang có hiệu lực",
        ],
    },
    # Hai intent dưới mang nghĩa loại trừ -> luôn đứng một mình
    "chitchat": {
        "open": [
            "chào bạn, hôm nay thời tiết thế nào",
            "chào bạn, dạo này có gì mới không",
            "cảm ơn bạn nhiều nhé",
            "bạn có khỏe không, cuối tuần này bạn làm gì",
        ],
        "join": [],
    },
    "general": {
        "open": [
            "cho mình hỏi về {topic} thì cần lưu ý những quy định gì",
            "{topic} hiện nay có những quy định gì cần biết",
            "mình đang tìm hiểu về {topic}, cần nắm những gì",
        ],
        "join": [],
    },
}

COMBINABLE = [
    "legal_query", "document_retrieval", "document_relationship",
    "comparative_analysis", "external_analysis", "stats_summary",
]
STANDALONE = ["chitchat", "general"]

OPENERS = ["Cho tôi hỏi, ", "Cho mình hỏi, ", "Xin hỏi, ", "", "", ""]
CONNECTORS = [", đồng thời ", ", và ", ". Ngoài ra ", ", bên cạnh đó "]
TAILS = [
    "?", "?", " vì tôi đang cần nắm rõ để làm thủ tục?",
    " ạ?", " giúp tôi với?", " vì công ty tôi đang vướng vấn đề này?",
]

SYSTEM_PROMPT = None  # nạp từ dữ liệu gốc để giữ nguyên 100%


def make_doc(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.45:
        return f"{rng.randint(1, 280)}/{rng.randint(2019, 2025)}/NĐ-CP"
    if kind < 0.80:
        return f"{rng.randint(1, 150)}/{rng.randint(2019, 2025)}/{rng.choice(MINISTRIES)}"
    if kind < 0.90:
        return f"{rng.randint(1, 99)}/{rng.randint(2014, 2024)}/QH{rng.randint(13, 15)}"
    if kind < 0.96:
        return f"{rng.randint(1, 90)}/VBHN-{rng.choice(['VPQH', 'BTC', 'BYT', 'BGDĐT'])}"
    return f"{rng.randint(1, 60)}/{rng.randint(2020, 2025)}/QĐ-TTg"


def fill(template: str, doc: str, topic: str, art: int) -> str:
    return template.format(doc=doc, art=art, topic=topic)


def build_question(intents: list[str], rng: random.Random) -> str:
    """Ghép các mệnh đề intent thành 1 câu hỏi. Mỗi intent = 1 mệnh đề thật."""
    doc = make_doc(rng)
    topic = rng.choice(TOPICS)
    art = rng.randint(1, 140)          # cùng một điều xuyên suốt câu hỏi
    parts = [fill(rng.choice(CLAUSES[intents[0]]["open"]), doc, topic, art)]
    for intent in intents[1:]:
        pool = CLAUSES[intent]["join"] or CLAUSES[intent]["open"]
        parts.append(fill(rng.choice(pool), doc, topic, art))

    text = parts[0]
    for part in parts[1:]:
        text += rng.choice(CONNECTORS) + part
    text = rng.choice(OPENERS) + text
    return text[0].upper() + text[1:] + rng.choice(TAILS)


def pick_intents(n: int, rng: random.Random) -> list[str]:
    if n == 1:
        # 1 nhãn: cho phép cả nhãn loại trừ (chitchat/general)
        return [rng.choice(COMBINABLE + STANDALONE + STANDALONE)]
    return rng.sample(COMBINABLE, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data_train/base_legal_7b_full.jsonl"),
                    help="File gốc, dùng để lấy nguyên văn system prompt của V3.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500, help="Tổng số mẫu sinh ra.")
    ap.add_argument("--dist", default="30,33,26,11",
                    help="Phân bố %% số mẫu có 1,2,3,4 nhãn.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # lấy nguyên văn system prompt của V3 để không đổi định dạng bài toán
    system_prompt = None
    with args.source.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("group") == "V3":
                system_prompt = row["messages"][0]["content"]
                break
    if system_prompt is None:
        raise SystemExit("Không tìm thấy mẫu V3 nào trong file nguồn.")

    weights = [int(x) for x in args.dist.split(",")]
    counts = [round(args.n * w / sum(weights)) for w in weights]
    counts[-1] = args.n - sum(counts[:-1])

    rows, idx, seen = [], 0, set()
    for n_labels, count in zip([1, 2, 3, 4], counts):
        made = 0
        attempts = 0
        while made < count and attempts < count * 60:
            attempts += 1
            intents = pick_intents(n_labels, rng)
            question = build_question(intents, rng)
            if question in seen:          # tránh sinh trùng câu hỏi
                continue
            seen.add(question)
            made += 1
            idx += 1
            rows.append({
                "id": f"V3F-{idx:04d}",
                "group": "V3",
                "task_family": "benchmark",
                "task_name": "user_intent_understanding",
                "benchmark_alignment": "VietLegal-2.5",
                "slice": f"{n_labels}_label",
                "difficulty": {1: "easy", 2: "medium", 3: "medium", 4: "hard"}[n_labels],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Earlier turns:\n(none)\n\nCurrent turn:\n{question}"},
                    {"role": "assistant",
                     "content": json.dumps({"intents": sorted(intents)}, ensure_ascii=False,
                                           separators=(",", ":"))},
                ],
            })

    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Đã sinh {len(rows)} mẫu -> {args.output}")


if __name__ == "__main__":
    main()
