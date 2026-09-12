#!/usr/bin/env python3
"""Sinh lại dữ liệu train task 2.5 (nhóm V3 - user_intent_understanding) bằng Claude API.

BỆNH ĐÃ CHẨN ĐOÁN Ở BỘ v3 (đo trên 1000 mẫu train):
  - 91.6% câu 4 nhãn / 92.4% câu 3 nhãn được ghép MÁY MÓC: mỗi intent một mệnh đề
    riêng nối bằng ", đồng thời", ". Ngoài ra", ". Còn nữa"...
  - Độ dài tăng tuyến tính theo số nhãn (16 -> 25 -> 35 -> 44 từ), tức mỗi intent
    cộng thêm đúng một mệnh đề.
  => Model chỉ học quy tắc "đếm mệnh đề = đếm intent". Gặp câu hỏi thật (ý định đan
     vào nhau, câu ngắn, câu cụt) nó chỉ nhận ra 1 ý: đo được 1.16 nhãn/câu so với
     2.19 nhãn/câu cần có, recall 32%.

NGUYÊN TẮC GÁN NHÃN — GIỮ NGUYÊN như bộ hiện tại, KHÔNG copy benchmark:
  - Mỗi nhãn ứng với một ý định THẬT có trong câu hỏi, không gán thừa.
  - Câu quá mơ hồ/thiếu thông tin: dùng `general` (có liên quan pháp luật) hoặc
    `chitchat` (không liên quan) — đây là áp dụng đúng taxonomy sẵn có cho input mơ
    hồ, KHÔNG phải kiểu "liệt kê mọi hướng xử lý khả dĩ" của benchmark.

CHỐNG OVERFIT BENCHMARK:
  1. Script này KHÔNG đọc file benchmark nào (vlegal_task_2_5_test.jsonl) — kể cả
     làm ví dụ few-shot.
  2. Pool chủ đề viết tay độc lập, không theo phân bố lĩnh vực của benchmark.
  3. Sau khi sinh, chạy scripts/audit_generated_2_5.py để đo rò rỉ n-gram với
     benchmark và kiểm tra baseline bag-of-words.
"""

import argparse
import json
import os
import random
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

INTENTS = [
    "chitchat", "comparative_analysis", "document_relationship", "document_retrieval",
    "external_analysis", "general", "legal_query", "stats_summary",
]

# Cụm nối liệt kê máy móc của bộ v3 — cấm tuyệt đối, validate lại sau khi sinh.
BANNED_CONNECTORS = [
    "đồng thời", "ngoài ra", "bên cạnh đó", "thêm nữa", "luôn tiện",
    "còn nữa", "với lại", "tiện thể", "nhân tiện",
]

# Pool chủ đề viết tay, phủ rộng lĩnh vực pháp luật VN — KHÔNG lấy từ benchmark.
TOPICS = [
    "hợp đồng lao động và trợ cấp thôi việc", "bảo hiểm xã hội tự nguyện",
    "thừa kế theo di chúc", "ly hôn và phân chia tài sản chung",
    "cấp sổ đỏ cho đất khai hoang", "tranh chấp ranh giới đất liền kề",
    "thuế thu nhập cá nhân từ chuyển nhượng bất động sản", "hoàn thuế giá trị gia tăng",
    "đăng ký kinh doanh hộ cá thể", "giải thể doanh nghiệp", "phá sản và thứ tự ưu tiên thanh toán",
    "góp vốn bằng tài sản trí tuệ", "bảo hộ kiểu dáng công nghiệp", "tranh chấp tên miền",
    "vi phạm bản quyền phần mềm", "bảo vệ dữ liệu cá nhân của khách hàng",
    "an toàn thực phẩm trong bếp ăn tập thể", "giấy phép hành nghề dược",
    "khám chữa bệnh trái tuyến và bảo hiểm y tế", "hiến mô tạng",
    "xử phạt nồng độ cồn khi lái xe", "đăng kiểm xe cải tạo",
    "tai nạn lao động và bồi thường", "an toàn phòng cháy chữa cháy chung cư",
    "giấy phép xây dựng nhà ở riêng lẻ", "cải tạo chung cư cũ",
    "nhập khẩu hàng hoá qua thương mại điện tử", "thủ tục hải quan hàng quá cảnh",
    "quảng cáo thực phẩm chức năng", "khuyến mại và trách nhiệm với người tiêu dùng",
    "cho vay ngang hàng", "lãi suất và xử lý nợ xấu", "tiền mã hoá và nghĩa vụ khai báo",
    "chào bán chứng khoán riêng lẻ", "công bố thông tin của công ty đại chúng",
    "đấu thầu thuốc trong bệnh viện công", "mua sắm tài sản công",
    "khiếu nại quyết định hành chính", "thi hành án dân sự", "án phí và lệ phí toà",
    "trách nhiệm hình sự của pháp nhân thương mại", "phòng chống rửa tiền",
    "nghĩa vụ quân sự và tạm hoãn", "hộ tịch, căn cước và cư trú",
    "nuôi con nuôi có yếu tố nước ngoài", "lao động nước ngoài làm việc tại Việt Nam",
    "đánh giá tác động môi trường dự án", "xả thải vượt quy chuẩn",
    "khai thác khoáng sản làm vật liệu xây dựng", "bảo vệ rừng phòng hộ",
    "trợ cấp thất nghiệp", "tiền lương làm thêm giờ", "kỷ luật lao động sa thải",
    "giáo dục nghề nghiệp và liên kết đào tạo", "tuyển sinh đầu cấp",
]

REGISTERS = [
    ("formal", "văn phong trang trọng, đầy đủ chủ ngữ vị ngữ"),
    ("casual", "văn phong nói chuyện đời thường, có thể dùng 'mình', 'ạ', 'nhỉ'"),
    ("terse", "cụt lủn, lược bỏ chủ ngữ, như gõ vội"),
    ("worried", "giọng lo lắng, kể tình huống cá nhân của người hỏi"),
    ("professional", "giọng người làm nghề (kế toán, nhân sự, luật sư nội bộ)"),
]

# Các dạng câu mà bộ v3 hoàn toàn thiếu.
SHAPES = [
    ("natural_multi", "Nhiều ý định ĐAN vào nhau trong MỘT câu liền mạch, tuyệt đối "
                      "không tách thành các mệnh đề liệt kê."),
    ("implicit", "Ý định thể hiện NGẦM qua tình huống, không nói thẳng ('tôi cần biết X', "
                 "'cho tôi Y'). Người đọc phải suy ra."),
    ("embedded_clause", "Ý định phụ nằm trong mệnh đề phụ (bắt đầu bằng 'mà', 'vì', "
                        "'trong khi', 'do') chứ không phải câu riêng."),
    ("short", "RẤT NGẮN, 3-8 từ, vẫn đủ để xác định ý định."),
    ("typo", "Có 2-3 lỗi chính tả/gõ vội tự nhiên (thiếu dấu, gõ nhầm phím gần nhau)."),
    ("no_diacritics", "Viết HOÀN TOÀN KHÔNG DẤU, như gõ vội trên điện thoại."),
    ("doc_ref", "Xoay quanh một số hiệu văn bản hoặc tên luật cụ thể."),
    ("greeting_mixed", "Mở đầu bằng lời chào/cảm ơn rồi hỏi tiếp trong cùng đoạn."),
    ("vague", "MƠ HỒ, thiếu thông tin, không đủ để xác định ý định cụ thể nào."),
]

SYSTEM = """Bạn là chuyên gia xây dựng dữ liệu huấn luyện cho hệ thống phân loại ý định (intent) \
của trợ lý pháp luật Việt Nam. Bạn viết ra các câu hỏi mà NGƯỜI DÙNG THẬT sẽ gõ vào hệ thống.

Danh sách 8 intent và định nghĩa:
- chitchat: không liên quan pháp luật (chào hỏi, cảm ơn, tán gẫu, off-topic)
- comparative_analysis: so sánh nội dung giữa hai văn bản/điều khoản/thời kỳ
- document_relationship: quan hệ giữa các văn bản (sửa đổi, bổ sung, thay thế, hướng dẫn, căn cứ, còn hiệu lực hay không)
- document_retrieval: xin toàn văn / bản gốc / file / link của văn bản
- external_analysis: tác động kinh tế - xã hội, ảnh hưởng thực tế, xu hướng, phản ứng thị trường
- general: câu hỏi tổng quát có liên quan pháp luật nhưng chưa thuộc intent cụ thể nào, hoặc quá mơ hồ để xác định
- legal_query: hỏi nội dung cụ thể của quy định (điều kiện, thủ tục, mức phạt, quyền/nghĩa vụ, định nghĩa)
- stats_summary: thống kê SỐ LƯỢNG văn bản/quy định, liệt kê danh sách văn bản

QUY TẮC GÁN NHÃN (rất quan trọng):
- Mỗi nhãn phải ứng với một ý định THẬT SỰ có trong câu. Không gán thừa nhãn chỉ vì chủ đề gần.
- Hỏi "được nghỉ bao nhiêu ngày phép" là legal_query (hỏi nội dung luật), KHÔNG phải stats_summary.
  stats_summary chỉ khi đếm SỐ LƯỢNG VĂN BẢN/QUY ĐỊNH.
- Câu quá mơ hồ không xác định được ý định cụ thể: gán đúng một nhãn `general` (nếu có hơi hướng
  pháp luật) hoặc `chitchat` (nếu hoàn toàn không liên quan).

QUY TẮC VIẾT CÂU (tuyệt đối tuân thủ):
- CẤM dùng các cụm nối liệt kê: "đồng thời", "ngoài ra", "bên cạnh đó", "thêm nữa", "luôn tiện",
  "còn nữa", "với lại", "tiện thể", "nhân tiện".
- CẤM viết kiểu ghép cơ học "ý A. Ngoài ra ý B. Còn nữa ý C" — đây chính là lỗi cần tránh.
- Khi câu có nhiều ý định, chúng phải hoà vào nhau tự nhiên như người thật viết: lồng trong mệnh đề
  phụ, dùng liên từ thường ("mà", "vì", "nếu", "khi", "để"), hoặc ngầm định.
- Đa dạng độ dài. Câu nhiều ý định KHÔNG nhất thiết phải dài hơn câu một ý định.

Chỉ trả về JSON hợp lệ, không giải thích gì thêm."""

USER_TEMPLATE = """Viết {k} câu hỏi tiếng Việt khác nhau cho hệ thống trợ lý pháp luật.

Yêu cầu cho lô này:
- Chủ đề gợi ý (dùng linh hoạt, có thể biến tấu): {topic}
- Văn phong: {register_desc}
- Dạng câu: {shape_desc}
- Mỗi câu phải mang ĐÚNG các intent sau, không thừa không thiếu: {intents}

Trả về JSON đúng dạng:
{{"items": [{{"question": "...", "intents": [{intents_json}]}}]}}

Nhắc lại: cấm mọi cụm nối liệt kê ("đồng thời", "ngoài ra", "còn nữa", "bên cạnh đó", ...). \
Các ý định phải đan tự nhiên trong câu."""


def strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def has_banned_connector(q: str) -> bool:
    low = strip_accents(q).lower()
    return any(strip_accents(c).lower() in low for c in BANNED_CONNECTORS)


def pick_spec(rng: random.Random) -> dict:
    """Chọn (số nhãn, bộ intent, shape, register, topic) cho một lô."""
    shape_name, shape_desc = rng.choice(SHAPES)

    if shape_name == "vague":
        n = 1
        intents = [rng.choice(["general", "chitchat"])]
    else:
        # phân bố số nhãn: nghiêng về 1-2 như truy vấn thật, vẫn đủ mẫu 3-4 nhãn
        n = rng.choices([1, 2, 3, 4], weights=[34, 33, 22, 11], k=1)[0]
        core = [i for i in INTENTS if i != "chitchat"]
        if n == 1:
            intents = [rng.choice(INTENTS)]
        else:
            intents = rng.sample(core, n)
            # chitchat chỉ ghép khi câu có lời chào/cảm ơn
            if shape_name == "greeting_mixed" and rng.random() < 0.6:
                intents = rng.sample(core, n - 1) + ["chitchat"]

    register_name, register_desc = rng.choice(REGISTERS)
    return {
        "n_labels": len(intents),
        "intents": sorted(intents),
        "shape": shape_name,
        "shape_desc": shape_desc,
        "register": register_name,
        "register_desc": register_desc,
        "topic": rng.choice(TOPICS),
    }


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
    ap.add_argument("--n", type=int, default=1000, help="Tổng số mẫu cần sinh.")
    ap.add_argument("--per-call", type=int, default=5, help="Số câu mỗi lần gọi API.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=None, help="Mặc định lấy từ .env, sửa về ID hợp lệ.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-source", type=Path,
                    default=Path("data-finetune-v2/data_finetune_v2_train.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("data-finetune-v2/data_finetune_v2_v3_regen.jsonl"))
    args = ap.parse_args()

    load_dotenv()
    model = args.model or "claude-sonnet-4-6"
    client = anthropic.Anthropic()

    # system prompt V3 lấy nguyên văn từ data hiện tại để không đổi định dạng bài toán
    system_prompt = None
    with args.prompt_source.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("group") == "V3":
                system_prompt = row["messages"][0]["content"]
                break
    if system_prompt is None:
        raise SystemExit("Không tìm thấy mẫu V3 trong file prompt-source.")

    rng = random.Random(args.seed)
    n_calls = (args.n + args.per_call - 1) // args.per_call
    specs = [pick_spec(rng) for _ in range(int(n_calls * 1.25))]  # dư để bù mẫu bị loại

    lock = threading.Lock()
    collected: list[dict] = []
    seen: set[str] = set()
    stats = {"calls": 0, "api_err": 0, "parse_err": 0, "rejected_connector": 0,
             "rejected_dup": 0, "in_tok": 0, "out_tok": 0}

    def run_one(spec: dict) -> None:
        if len(collected) >= args.n:
            return
        user = USER_TEMPLATE.format(
            k=args.per_call, topic=spec["topic"], register_desc=spec["register_desc"],
            shape_desc=spec["shape_desc"], intents=", ".join(spec["intents"]),
            intents_json=", ".join(f'"{i}"' for i in spec["intents"]),
        )
        try:
            resp = client.messages.create(
                model=model, max_tokens=2000,
                system=system_prompt + "\n\n---\n\n" + SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
        except Exception:
            with lock:
                stats["api_err"] += 1
            return

        text = "".join(b.text for b in resp.content if b.type == "text")
        obj = extract_json(text)
        with lock:
            stats["calls"] += 1
            stats["in_tok"] += resp.usage.input_tokens
            stats["out_tok"] += resp.usage.output_tokens
        if not obj or not isinstance(obj.get("items"), list):
            with lock:
                stats["parse_err"] += 1
            return

        for item in obj["items"]:
            q = str(item.get("question", "")).strip()
            got = item.get("intents")
            if not q or not isinstance(got, list):
                continue
            got = sorted({str(x).strip() for x in got})
            if got != spec["intents"] or any(g not in INTENTS for g in got):
                continue
            if has_banned_connector(q):
                with lock:
                    stats["rejected_connector"] += 1
                continue
            key = strip_accents(q).lower().strip(" ?.!")
            with lock:
                if key in seen:
                    stats["rejected_dup"] += 1
                    continue
                if len(collected) >= args.n:
                    return
                seen.add(key)
                collected.append({"question": q, "intents": got, **{
                    k: spec[k] for k in ("shape", "register", "n_labels")}})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, s) for s in specs]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 20 == 0:
                print(f"  ... {len(collected)}/{args.n} mẫu  ({i}/{len(futures)} lượt gọi)", flush=True)

    rng.shuffle(collected)
    collected = collected[: args.n]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for idx, item in enumerate(collected, 1):
            row = {
                "id": f"V3G-{idx:04d}",
                "group": "V3",
                "task_family": "benchmark",
                "task_name": "user_intent_understanding",
                "benchmark_alignment": "VietLegal-2.5",
                "slice": item["shape"],
                "difficulty": {1: "easy", 2: "medium", 3: "medium", 4: "hard"}[item["n_labels"]],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"Earlier turns:\n(none)\n\nCurrent turn:\n{item['question']}"},
                    {"role": "assistant",
                     "content": json.dumps({"intents": item["intents"]}, ensure_ascii=False,
                                           separators=(",", ":"))},
                ],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    cost = stats["in_tok"] / 1e6 * 3 + stats["out_tok"] / 1e6 * 15
    print(f"\nĐã sinh {len(collected)} mẫu -> {args.output}")
    print(f"Lượt gọi API: {stats['calls']} (lỗi API {stats['api_err']}, lỗi parse {stats['parse_err']})")
    print(f"Loại bỏ: {stats['rejected_connector']} câu dùng từ nối cấm, {stats['rejected_dup']} câu trùng")
    print(f"Token: {stats['in_tok']:,} in / {stats['out_tok']:,} out  (~${cost:.2f} với {model})")


if __name__ == "__main__":
    main()
