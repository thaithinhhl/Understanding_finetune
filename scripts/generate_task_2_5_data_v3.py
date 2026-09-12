#!/usr/bin/env python3
"""Sinh dữ liệu train v3 cho task 2.5 (nhóm V3 - user_intent_understanding).

Khắc phục lỗi phát hiện ở v2: mỗi intent chỉ gắn với ĐÚNG MỘT cụm từ khoá
("bao nhiêu"->stats_summary, "toàn văn"->document_retrieval, "so sánh"->
comparative_analysis, ...) với precision 100% và ZERO phản ví dụ. Đo bằng
baseline bag-of-words (không hiểu ngữ nghĩa, chỉ khớp từ) đạt F1 86.6% trên
data v2 -> chứng minh model có thể "giải" bằng tra từ điển, không cần hiểu.

v3 sửa bằng 2 cơ chế, cả hai đều bắt buộc với MỌI intent:

1. Mỗi intent có NHIỀU cụm diễn đạt không dùng chung 1 từ khoá (ít nhất 4 "họ
   từ vựng" khác nhau), để không có một token nào bao phủ >30% số câu của intent.
2. PHẢN VÍ DỤ rõ ràng: từ khoá đặc trưng của intent A xuất hiện trong câu
   KHÔNG mang nhãn A, vì mang nghĩa khác trong ngữ cảnh khác. Ví dụ:
   "bao nhiêu" trong "Điều 5 quy định nghỉ phép bao nhiêu ngày" là legal_query
   (hỏi nội dung luật), không phải stats_summary (hỏi số lượng văn bản).

Nguyên tắc gán nhãn giữ nguyên: mỗi nhãn ứng với một ý THẬT có trong câu hỏi.
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

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
    "hộ tịch và căn cước", "cư trú và tạm trú", "sở hữu nhà ở của người nước ngoài",
    "trợ cấp thôi việc", "kiểm định chất lượng giáo dục", "đăng kiểm xe cơ giới",
    "quảng cáo thương mại", "thương mại điện tử", "giải quyết tranh chấp lao động",
    "nuôi con nuôi", "phá sản doanh nghiệp", "chứng thực chữ ký",
]

# Đại lượng cụ thể để hỏi "bao nhiêu X" mang nghĩa NỘI DUNG luật (legal_query),
# KHÔNG phải hỏi số lượng văn bản (stats_summary) -> phản ví dụ cho từ "bao nhiêu".
QUANTITIES = [
    "ngày phép năm", "ngày nghỉ thai sản", "tháng thử việc", "% thuế suất",
    "triệu đồng tiền phạt", "năm tù", "ngày để nộp hồ sơ", "% cổ phần tối đa",
    "tháng để khiếu nại", "triệu đồng vốn điều lệ tối thiểu", "ngày công khai thông tin",
]

# ------------------------------------------------------ mệnh đề theo từng intent
# Mỗi intent có >=4 "họ từ vựng" (families) KHÔNG dùng chung từ khoá, để không có
# 1 token nào phủ >30% số câu. "join" dùng khi ghép sau mệnh đề khác.
CLAUSES = {
    "legal_query": {
        "families": [
            # họ 1: hỏi nội dung điều luật
            ["cho tôi biết nội dung Điều {art} của văn bản {doc} quy định gì về {topic}",
             "Điều {art} trong văn bản {doc} nói gì về {topic}",
             "nội dung Điều {art} quy định thế nào"],
            # họ 2: hỏi quy trình/điều kiện (không dùng "Điều", "quy định")
            ["muốn làm thủ tục về {topic} thì cần những giấy tờ gì",
             "điều kiện để được {topic} là gì",
             "trình tự thực hiện {topic} ra sao"],
            # họ 3: hỏi trực tiếp khái niệm
            ["{topic} được hiểu như thế nào theo luật",
             "định nghĩa pháp lý của {topic} là gì",
             "{topic} là gì theo quy định hiện hành"],
            # họ 4: phản ví dụ có chứa "bao nhiêu" nhưng KHÔNG phải stats_summary
            ["theo Điều {art} thì được nghỉ bao nhiêu {qty}",
             "mức xử phạt là bao nhiêu {qty} cho hành vi này",
             "thời hạn giải quyết là bao nhiêu {qty}"],
            # họ 5: phản ví dụ chứa từ khoá của các intent khác nhưng vẫn là legal_query thuần
            ["tôi đã đọc toàn văn rồi nhưng vẫn chưa rõ Điều {art} nói gì về {topic}",
             "so sánh với các nước khác thì luật về {topic} có gì đặc biệt",
             "hành vi vi phạm về {topic} tác động thế nào đến quyền lợi người trong cuộc theo luật",
             "cần lưu ý điều kiện gì để được miễn trừ theo Điều {art}"],
            # họ 5b: lặp lại họ 5 để tăng tần suất phản ví dụ (giảm precision của các từ khoá)
            ["đọc toàn văn xong vẫn không hiểu Điều {art} áp dụng cho {topic} thế nào",
             "quy định này tác động ra sao đến nghĩa vụ cụ thể của tôi theo Điều {art}",
             "lưu ý là cần đáp ứng điều kiện gì theo Điều {art} về {topic}"],
        ],
        "join": [
            "Điều {art} nêu cụ thể những gì",
            "điều đó quy định thế nào về {topic}",
            "còn về điều kiện thực hiện thì sao",
            "cụ thể mức áp dụng là bao nhiêu {qty}",
            "tôi đã xem toàn văn nhưng vẫn chưa rõ điểm này",
            "hành vi đó tác động thế nào đến quyền lợi cụ thể của tôi theo luật",
            "cần lưu ý điều kiện cụ thể gì để được áp dụng",
            "so sánh ra thì Điều {art} áp dụng cho {topic} có gì đặc thù",
        ],
    },
    "document_retrieval": {
        "families": [
            ["cho tôi xin toàn văn văn bản {doc}", "xin vui lòng cung cấp toàn văn văn bản {doc}"],
            ["gửi tôi file PDF của văn bản về {topic}", "cho xin bản gốc văn bản {doc}"],
            ["có link tải văn bản {doc} không", "văn bản {doc} tải ở đâu vậy"],
            ["cho mình xin nguyên văn quy định về {topic}", "đính kèm giúp mình văn bản {doc} nhé"],
        ],
        "join": [
            "cho tôi xin toàn văn văn bản đó luôn",
            "gửi kèm file luôn nhé",
            "cho xin bản đầy đủ luôn",
            "đính kèm giúp tôi văn bản gốc",
        ],
    },
    "document_relationship": {
        "families": [
            ["Điều {art} văn bản {doc} có bị sửa đổi hay thay thế bởi văn bản nào không",
             "văn bản {doc} có văn bản nào hướng dẫn thi hành không"],
            ["văn bản {doc} còn hiệu lực không hay đã hết hiệu lực",
             "quy định này còn áp dụng được không"],
            ["có nghị định nào kế thừa văn bản {doc} không",
             "văn bản nào đã bãi bỏ nội dung này chưa"],
            ["mối liên hệ giữa văn bản {doc} và các văn bản liên quan là gì",
             "văn bản {doc} có phụ thuộc vào văn bản gốc nào không"],
        ],
        "join": [
            "văn bản đó có bị thay thế chưa",
            "có bị bãi bỏ chưa",
            "còn hiệu lực không",
            "có văn bản nào hướng dẫn thêm không",
        ],
    },
    "comparative_analysis": {
        "families": [
            ["so sánh Điều {art} của văn bản {doc} với quy định trước đó về {topic}",
             "Điều {art} về {topic} có gì khác so với quy định cũ"],
            ["điểm khác biệt giữa quy định mới và quy định cũ về {topic} là gì",
             "trước và sau khi sửa đổi thì {topic} thay đổi ra sao"],
            ["đối chiếu quy định hiện hành với quy định trước đây về {topic}",
             "quy định về {topic} qua các thời kỳ có gì đổi khác"],
            ["giữa hai văn bản này thì cái nào ưu việt hơn về {topic}",
             "điểm tiến bộ hơn của quy định mới về {topic} là gì"],
        ],
        "join": [
            "so với trước đó thì khác nhau ra sao",
            "đối chiếu thì thấy khác gì",
            "có gì tiến bộ hơn so với trước",
            "thay đổi cụ thể là gì",
        ],
    },
    "external_analysis": {
        "families": [
            ["quy định về {topic} thay đổi đã tác động thế nào đến kinh tế - xã hội",
             "những thay đổi trong quy định về {topic} ảnh hưởng ra sao đến thị trường"],
            ["quy định mới về {topic} kéo theo hệ lụy gì cho doanh nghiệp",
             "hiệu ứng thực tế của chính sách {topic} đến người dân là gì"],
            ["việc siết chặt quy định {topic} sẽ làm thị trường biến động thế nào",
             "sau khi áp dụng, {topic} khiến chi phí tuân thủ tăng hay giảm"],
            ["các bên liên quan phản ứng thế nào trước thay đổi về {topic}",
             "dư luận đánh giá ra sao về chính sách {topic} mới"],
        ],
        "join": [
            "tác động thực tế ra sao",
            "kéo theo hệ lụy gì không",
            "ảnh hưởng thế nào đến người dân và doanh nghiệp",
            "phản ứng của thị trường thế nào",
        ],
    },
    "stats_summary": {
        "families": [
            ["xin hỏi hiện có bao nhiêu quy định về {topic}", "cho hỏi có bao nhiêu văn bản đang quy định về {topic}"],
            ["cho tôi số liệu tổng hợp về các văn bản liên quan {topic}",
             "tổng số văn bản đang điều chỉnh {topic} là mấy"],
            ["đếm giúp tôi có mấy nghị định về {topic}", "liệt kê xem có mấy văn bản về {topic}"],
            ["danh sách đầy đủ các quy định về {topic} gồm mấy văn bản",
             "con số cụ thể văn bản hiện hành về {topic} là bao nhiêu"],
        ],
        "join": [
            "tổng cộng có mấy văn bản như vậy",
            "con số cụ thể là bao nhiêu",
            "liệt kê giúp tôi luôn",
            "tổng hợp số lượng giúp tôi",
        ],
    },
    "general": {
        "families": [
            ["cho mình hỏi về {topic} thì cần lưu ý những quy định gì",
             "{topic} hiện nay có những quy định gì cần biết"],
            ["mình đang tìm hiểu về {topic}, nên bắt đầu từ đâu",
             "có kinh nghiệm gì chia sẻ về {topic} không"],
            ["quy trình chung khi làm việc liên quan {topic} là gì",
             "tổng quan về {topic} thì gồm những gì"],
            ["mình mù mờ về {topic} quá, giải thích giúp mình với",
             "{topic} nói chung thì phức tạp không"],
        ],
        "join": [
            "nói chung về {topic} thì cần biết gì thêm",
            "tổng quan thì sao",
            "có gì cần lưu ý thêm không",
            "còn nói riêng về khía cạnh này thì {topic} thế nào",
        ],
    },
}

# chitchat: đa dạng, không chỉ dùng "chào" -> giảm trigger đơn nhất.
CHITCHAT_OPENERS = [
    "Chào bạn, ", "Hi bạn, ", "Này bạn, ", "Bạn ơi, ", "Ê, ", "Alo, ",
    "Cho mình hỏi xíu nè, ", "",
]
CHITCHAT_CLOSERS = [
    " Cảm ơn bạn nhiều nhé!", " Thanks bạn nha!", " Bạn giúp mình với nhé!",
    " Chúc bạn ngày tốt lành!", " Merci bạn!",
]
CHITCHAT_ONLY = [
    "hôm nay thời tiết thế nào", "dạo này có gì mới không", "cảm ơn bạn nhiều nhé",
    "bạn có khỏe không", "bạn tên gì vậy", "bạn ăn cơm chưa", "cuối tuần này làm gì đó",
    "bạn là ai vậy", "chúc bạn ngày mới vui vẻ",
]

CORE_INTENTS = [
    "legal_query", "document_retrieval", "document_relationship",
    "comparative_analysis", "external_analysis", "stats_summary", "general",
]
REGISTERS_CONNECTORS = [
    ", đồng thời ", ", và ", ". Ngoài ra ", ", bên cạnh đó ", ". Thêm nữa ",
    ", luôn tiện ", ". Còn nữa, ", ", với lại ",
]
TAILS = ["?", "?", " ạ?", " nhỉ?", " vậy?", " nha?", "", " giúp mình với?"]
OPENERS = ["Cho tôi hỏi, ", "Cho mình hỏi, ", "Xin hỏi, ", "Xin chào, cho tôi hỏi, ", "", "", "", ""]


def strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def unify_pronoun(text: str, rng: random.Random) -> str:
    target = rng.choice(["tôi", "mình"])
    other = "mình" if target == "tôi" else "tôi"
    return re.sub(rf"\b{other}\b", target, text)


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


def fill(template: str, doc: str, topic: str, art: int, qty: str) -> str:
    return template.format(doc=doc, art=art, topic=topic, qty=qty)


def pick_family_clause(intent: str, rng: random.Random) -> str:
    fam = rng.choice(CLAUSES[intent]["families"])
    return rng.choice(fam)


def pick_intents(rng: random.Random) -> list[str]:
    # phân bổ đều số nhãn (25% mỗi mức 1-4), không lệch theo bất kỳ bộ test nào
    n = rng.choices([1, 2, 3, 4], weights=[25, 25, 25, 25], k=1)[0]
    # chitchat được nâng tần suất lên ngang các intent nội dung khác (~30%)
    # thay vì chỉ 16% như trước, để 8 nhãn có tần suất marginal gần đều nhau
    has_chitchat = rng.random() < 0.32
    if has_chitchat:
        if n == 1:
            return ["chitchat"]
        core = rng.sample(CORE_INTENTS, n - 1)
        return sorted(core + ["chitchat"])
    return sorted(rng.sample(CORE_INTENTS, n))


def build_core(core: list[str], rng: random.Random) -> str:
    doc = make_doc(rng) if rng.random() < 0.35 else "này"
    topic = rng.choice(TOPICS)
    art = rng.randint(1, 140)
    qty = rng.choice(QUANTITIES)

    order = core[:]
    rng.shuffle(order)
    first = fill(pick_family_clause(order[0], rng), doc, topic, art, qty)
    parts = [first]
    for intent in order[1:]:
        parts.append(fill(rng.choice(CLAUSES[intent]["join"]), doc, topic, art, qty))

    text = parts[0]
    for part in parts[1:]:
        text += rng.choice(REGISTERS_CONNECTORS) + part
    text = rng.choice(OPENERS) + text
    text = text[0].upper() + text[1:] + rng.choice(TAILS)
    return unify_pronoun(text, rng)


def build_question(intents: list[str], rng: random.Random) -> str:
    core = [i for i in intents if i != "chitchat"]
    if not core:
        q = rng.choice(CHITCHAT_ONLY)
        return q[0].upper() + q[1:] + rng.choice(["?", "?", " nhỉ?", "!"])
    core_q = build_core(core, rng)
    if "chitchat" in intents:
        base = core_q[0].lower() + core_q[1:]
        q = rng.choice(CHITCHAT_OPENERS) + base
        if rng.random() < 0.45:
            q += rng.choice(CHITCHAT_CLOSERS)
    else:
        q = core_q
    if rng.random() < 0.08:
        q = strip_accents(q)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data_train/base_legal_7b_full.jsonl"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    system_prompt = None
    with args.source.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("group") == "V3":
                system_prompt = row["messages"][0]["content"]
                break
    if system_prompt is None:
        raise SystemExit("Không tìm thấy mẫu V3 nào trong file nguồn.")

    rows, idx, seen = [], 0, set()
    attempts = 0
    while len(rows) < args.n and attempts < args.n * 40:
        attempts += 1
        intents = pick_intents(rng)
        question = build_question(intents, rng)
        if question in seen:
            continue
        seen.add(question)
        idx += 1
        n_labels = len(intents)
        rows.append({
            "id": f"V3F3-{idx:04d}",
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
                 "content": json.dumps({"intents": intents}, ensure_ascii=False,
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
