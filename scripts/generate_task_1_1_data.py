#!/usr/bin/env python3
"""Sinh lại dữ liệu train task 1.1 (nhóm V2 - legal_entity_extraction) bằng Claude API.

BỆNH ĐÃ CHẨN ĐOÁN Ở BỘ V2 CŨ (data-finetune-v2/data_finetune_v2_train.jsonl, 1000 mẫu):
  - STATE_BODY có 544 lượt xuất hiện nhưng chỉ từ 6 giá trị duy nhất, lặp 80-101 lần
    mỗi giá trị ("Ủy ban nhân dân tỉnh Bắc Ninh", "Bộ Tài chính", "Công an tỉnh Nghệ
    An", "Sở Xây dựng TP Hà Nội", "Viện kiểm sát nhân dân TP Hồ Chí Minh", "Cục Thuế
    tỉnh Đồng Nai") => model học thuộc 6 chuỗi này thay vì học nhận diện cơ quan nhà
    nước nói chung. Gặp đoạn văn thật, model rò rỉ ("hallucinate") một trong 6 tên
    quen thuộc dù không có trong văn bản.
  - Thể loại văn bản gần như chỉ có MỘT khuôn: "Tòa án ... xét xử sơ thẩm bị cáo ...".
    Từ khoá chính luận/hành chính tổng quát (Chính phủ, Đảng, Quốc hội, Mặt trận Tổ
    quốc, Nhân dân) gần như vắng mặt (0-21/1000 mẫu).

NGUYÊN TẮC CHỐNG BIAS (bắt buộc, theo yêu cầu của user 2026-09-13):
  - Script này KHÔNG đọc bất kỳ file benchmark nào (data-benchmark-v2/, vlegal_task_*
    _test*.jsonl) — kể cả để lấy ví dụ few-shot hay tham khảo phân bố nhãn/thực thể.
    Data train phải tách biệt hoàn toàn khỏi benchmark để không bias/overfit.
  - Pool thể loại + pool giá trị thực thể viết tay độc lập, phủ rộng nhiều tỉnh/bộ/
    ngành/thể loại — không lấy từ benchmark.

CƠ CHẾ CHỐNG TRÙNG LẶP GIÁ TRỊ (khắc phục đúng lỗi đã tìm thấy):
  - Với mỗi loại thực thể "rủi ro cao" (STATE_BODY, ORGANIZATION, COURT, LAW,
    LOCATION), mỗi lượt gọi API được GÁN SẴN một giá trị cụ thể rút ra từ pool lớn
    theo vòng round-robin đã xáo trộn (không phải random.choice thuần) để đảm bảo
    mọi giá trị trong pool được dùng gần như đều nhau, không giá trị nào lấn át.
  - PERSON được sinh ngẫu nhiên từ tổ hợp họ+tên lớn (hàng trăm tổ hợp).
  - Sau khi sinh, validate: mọi entity.text phải xuất hiện NGUYÊN VĂN trong đoạn văn
    (đúng rule 1 của system prompt gốc) — loại bỏ mẫu không đạt.
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

ENTITY_TYPES = {
    "PERSON", "LEGAL_ROLE", "STATE_BODY", "ORGANIZATION", "COURT", "LAW",
    "LEGAL_DOCUMENT", "LEGAL_PROVISION", "CASE_ID", "DATE", "MONEY", "LOCATION",
}

# ---------------------------------------------------------------------------
# Pool giá trị thực thể — viết tay, độc lập với benchmark, phủ nhiều tỉnh/bộ/ngành
# để một giá trị không thể chiếm phần lớn số lượt như bộ V2 cũ.
# ---------------------------------------------------------------------------

PROVINCES = [
    "Bắc Ninh", "Nghệ An", "Hải Phòng", "Đà Nẵng", "Cần Thơ", "Thanh Hóa",
    "Quảng Ninh", "Bình Dương", "Khánh Hòa", "Lâm Đồng", "An Giang", "Kiên Giang",
    "Đắk Lắk", "Thái Nguyên", "Nam Định", "Quảng Nam", "Bình Định", "Phú Thọ",
    "Hà Tĩnh", "Vĩnh Phúc", "Bắc Giang", "Long An", "Tiền Giang", "Sóc Trăng",
    "Gia Lai", "Lạng Sơn", "Cà Mau", "Bến Tre", "Hưng Yên", "Ninh Bình",
]
CITIES_CENTRAL = ["TP Hà Nội", "TP Hồ Chí Minh", "TP Hải Phòng", "TP Đà Nẵng", "TP Cần Thơ"]

STATE_BODY_TEMPLATES = [
    "Ủy ban nhân dân tỉnh {p}", "Ủy ban nhân dân {p}",
    "Hội đồng nhân dân tỉnh {p}", "Công an tỉnh {p}", "Công an {p}",
    "Viện kiểm sát nhân dân tỉnh {p}", "Viện kiểm sát nhân dân {p}",
    "Cục Thuế tỉnh {p}", "Sở Xây dựng {p}", "Sở Kế hoạch và Đầu tư tỉnh {p}",
    "Sở Tài nguyên và Môi trường tỉnh {p}", "Sở Giao thông vận tải tỉnh {p}",
    "Sở Y tế tỉnh {p}", "Sở Giáo dục và Đào tạo tỉnh {p}",
    "Sở Nông nghiệp và Môi trường tỉnh {p}", "Thanh tra tỉnh {p}",
    "Chi cục Thi hành án dân sự {p}",
]
STATE_BODY_FIXED = [
    "Bộ Tài chính", "Bộ Công Thương", "Bộ Y tế", "Bộ Giáo dục và Đào tạo",
    "Bộ Nội vụ", "Bộ Xây dựng", "Bộ Giao thông vận tải",
    "Bộ Nông nghiệp và Môi trường", "Bộ Tư pháp", "Bộ Ngoại giao",
    "Bộ Quốc phòng", "Bộ Công an", "Bộ Kế hoạch và Đầu tư",
    "Bộ Lao động - Thương binh và Xã hội", "Bộ Văn hóa, Thể thao và Du lịch",
    "Quốc hội", "Ủy ban Thường vụ Quốc hội", "Chính phủ",
    "Thanh tra Chính phủ", "Kiểm toán Nhà nước", "Ngân hàng Nhà nước Việt Nam",
    "Tổng cục Thuế", "Tổng cục Hải quan", "Ủy ban Chứng khoán Nhà nước",
    "Bảo hiểm xã hội Việt Nam",
]

COURT_TEMPLATES = [
    "Tòa án nhân dân tỉnh {p}", "Tòa án nhân dân {p}",
    "Tòa án nhân dân huyện {p}", "Tòa Kinh tế Tòa án nhân dân {p}",
    "Tòa Hình sự Tòa án nhân dân {p}", "Tòa Dân sự Tòa án nhân dân {p}",
]
COURT_FIXED = [
    "Tòa án nhân dân tối cao", "Tòa án nhân dân cấp cao tại Hà Nội",
    "Tòa án nhân dân cấp cao tại Đà Nẵng", "Tòa án nhân dân cấp cao tại TP Hồ Chí Minh",
]

LAW_POOL = [
    "Luật Đất đai", "Luật Doanh nghiệp", "Bộ luật Lao động", "Bộ luật Dân sự",
    "Bộ luật Hình sự", "Bộ luật Tố tụng hình sự", "Bộ luật Tố tụng dân sự",
    "Luật Hôn nhân và Gia đình", "Luật Bảo hiểm xã hội", "Luật Giao thông đường bộ",
    "Luật Xây dựng", "Luật Đầu tư", "Luật Chứng khoán", "Luật Cạnh tranh",
    "Luật Sở hữu trí tuệ", "Hiến pháp", "Luật Nhà ở", "Luật Kinh doanh bất động sản",
    "Luật Bảo vệ môi trường", "Luật Phòng, chống tham nhũng", "Luật Ngân sách nhà nước",
    "Luật Quản lý thuế", "Luật Các tổ chức tín dụng", "Luật Giáo dục",
    "Luật Khám bệnh, chữa bệnh", "Luật Viên chức", "Luật Cán bộ, công chức",
    "Luật Xử lý vi phạm hành chính", "Luật Phá sản", "Luật Trọng tài thương mại",
]

LEGAL_DOC_TYPES = ["Nghị định", "Thông tư", "Nghị quyết", "Quyết định", "Công văn", "Chỉ thị"]
LEGAL_DOC_AGENCIES = ["CP", "BTC", "BYT", "BGDĐT", "BXD", "BCT", "TTg", "UBND", "NHNN", "BLĐTBXH"]

ORG_TEMPLATES = [
    "Công ty TNHH {n}", "Công ty Cổ phần {n}", "Công ty TNHH MTV {n}",
    "Tập đoàn {n}", "Ngân hàng TMCP {n}", "Hợp tác xã {n}",
]
ORG_NAME_WORDS = [
    "Thành Phát", "Đại Dương", "Việt Thắng", "Hoàng Long", "An Bình",
    "Sài Gòn Xanh", "Phú Cường", "Minh Khang", "Đông Á", "Tân Việt",
    "Sao Mai", "Kim Long", "Hòa Phát Miền Trung", "Sông Hồng", "Cửu Long",
    "Bảo Tín", "Nam Việt", "Gia Định", "Thịnh Vượng", "Hải Âu",
]

ORG_FIXED = [
    "Mặt trận Tổ quốc Việt Nam", "Hội Nông dân Việt Nam", "Hội Liên hiệp Phụ nữ Việt Nam",
    "Đoàn Thanh niên Cộng sản Hồ Chí Minh", "Tổng Liên đoàn Lao động Việt Nam",
    "Phòng Thương mại và Công nghiệp Việt Nam", "Hội Chữ thập đỏ Việt Nam",
]

LOCATION_POOL = (
    [f"tỉnh {p}" for p in PROVINCES] + CITIES_CENTRAL +
    ["quận Hải Châu", "quận Ninh Kiều", "huyện Củ Chi", "huyện Bình Chánh",
     "phường Dịch Vọng", "thị xã Sơn Tây", "khu công nghiệp Bắc Thăng Long",
     "xã Hòa Bắc", "thị trấn Sịa", "quận Lê Chân"]
)

FAMILY_NAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Ngô", "Dương", "Lý"]
MIDDLE_GIVEN = [
    "Văn An", "Thị Bích", "Minh Tuấn", "Thanh Hà", "Quốc Việt", "Thị Lan",
    "Hữu Nghĩa", "Ngọc Diệp", "Xuân Phúc", "Thị Mai", "Đình Khoa", "Thị Thu",
    "Anh Dũng", "Kim Ngân", "Thành Trung", "Thị Hồng", "Bảo Long", "Thùy Trang",
    "Công Danh", "Thị Nga",
]
LEGAL_ROLE_POOL = [
    "Chủ tịch Ủy ban nhân dân", "Phó Chủ tịch Ủy ban nhân dân", "Giám đốc Sở",
    "Phó Giám đốc Sở", "bị cáo", "nguyên đơn", "bị đơn", "người có quyền lợi, nghĩa vụ liên quan",
    "luật sư bào chữa", "kiểm sát viên", "thẩm phán chủ tọa", "Chánh án",
    "Trưởng phòng", "Tổng Giám đốc", "Giám đốc", "Kế toán trưởng",
    "Trưởng Công an", "điều tra viên", "công chứng viên", "Chấp hành viên",
]

CASE_ID_TYPES = ["HS-ST", "DS-ST", "HC-ST", "KDTM-ST", "LĐ-ST", "HS-PT"]

GENRES = [
    ("criminal_judgment", "Bản án hình sự sơ thẩm: mở đầu bằng ngày xét xử, tòa án, "
     "tội danh, tóm tắt hành vi phạm tội, số tiền liên quan (nếu có)."),
    ("civil_dispute", "Bản án/quyết định dân sự: tranh chấp hợp đồng, đất đai, thừa kế "
     "hoặc ly hôn giữa nguyên đơn và bị đơn, có thể nêu tài sản, số tiền tranh chấp."),
    ("administrative_decision", "Quyết định xử phạt vi phạm hành chính của một cơ quan "
     "nhà nước đối với cá nhân/tổ chức, nêu rõ căn cứ pháp lý, mức phạt."),
    ("govt_report", "Trích đoạn báo cáo/phát biểu chính luận của cơ quan nhà nước hoặc "
     "tổ chức chính trị - xã hội (Chính phủ, Quốc hội, Mặt trận Tổ quốc, Đảng) về một "
     "chủ trương, thành tựu hoặc nhiệm vụ — văn phong trang trọng, KHÔNG phải vụ án.")
    ,
    ("regulatory_news", "Tin tức về một văn bản pháp luật mới ban hành hoặc sửa đổi: cơ "
     "quan ban hành, tên/số hiệu văn bản, nội dung chính, ngày hiệu lực."),
    ("corporate_disclosure", "Công bố thông tin doanh nghiệp: một công ty/ngân hàng công "
     "bố quyết định, giao dịch, hoặc bị xử phạt bởi cơ quan quản lý (UBCKNN, NHNN...)."),
    ("labor_dispute", "Tranh chấp lao động: người lao động khiếu nại một công ty về sa "
     "thải, lương, bảo hiểm xã hội, có thể có vai trò của Sở LĐTBXH hoặc tòa án."),
    ("land_dispute", "Tranh chấp đất đai/xây dựng: giữa các hộ dân hoặc với chính quyền "
     "địa phương, liên quan Sở Xây dựng/Sở TNMT/UBND, có thể có quyết định cưỡng chế."),
    ("enforcement_notice", "Thông báo/quyết định thi hành án dân sự hoặc cưỡng chế thuế: "
     "cơ quan thi hành án hoặc cục thuế, đối tượng, số tiền, thời hạn."),
    ("meeting_announcement", "Tin về một phiên họp/kỳ họp của Quốc hội, Hội đồng nhân "
     "dân hoặc Chính phủ: nội dung thảo luận, nghị quyết thông qua, thời gian."),
]

DIFFICULTIES = ["easy", "medium", "hard"]

MAX_USES_PER_VALUE = 18  # trần dùng lặp cho 1 giá trị cụ thể trong pool rủi ro cao


class RoundRobinPool:
    """Cấp giá trị theo vòng đã xáo trộn, đảm bảo phân bổ gần như đều, có trần lặp."""

    def __init__(self, values: list[str], rng: random.Random, max_uses: int = MAX_USES_PER_VALUE):
        self.values = list(values)
        self.rng = rng
        self.max_uses = max_uses
        self.usage: dict[str, int] = {v: 0 for v in self.values}
        self._cycle = self._make_cycle()
        self._lock = threading.Lock()

    def _make_cycle(self):
        order = list(self.values)
        self.rng.shuffle(order)
        return order

    def next(self) -> str:
        with self._lock:
            for _ in range(len(self.values) * 3):
                if not self._cycle:
                    self._cycle = self._make_cycle()
                v = self._cycle.pop()
                if self.usage[v] < self.max_uses:
                    self.usage[v] += 1
                    return v
            # tất cả đã chạm trần (pool quá nhỏ so với --n) -> vẫn cấp, chỉ log lệch
            v = self.rng.choice(self.values)
            self.usage[v] += 1
            return v


def build_state_body_pool(rng: random.Random) -> list[str]:
    out = list(STATE_BODY_FIXED)
    for tmpl in STATE_BODY_TEMPLATES:
        for p in PROVINCES:
            out.append(tmpl.format(p=p))
    return out


def build_court_pool() -> list[str]:
    out = list(COURT_FIXED)
    for tmpl in COURT_TEMPLATES:
        for p in PROVINCES:
            out.append(tmpl.format(p=p))
    return out


def build_org_pool(rng: random.Random) -> list[str]:
    out = list(ORG_FIXED)
    for tmpl in ORG_TEMPLATES:
        for w in ORG_NAME_WORDS:
            out.append(tmpl.format(n=w))
    return out


def random_person(rng: random.Random) -> str:
    return f"{rng.choice(FAMILY_NAMES)} {rng.choice(MIDDLE_GIVEN)}"


def random_case_id(rng: random.Random) -> str:
    return f"{rng.randint(10, 999)}/{rng.randint(2022, 2026)}/{rng.choice(CASE_ID_TYPES)}"


def random_legal_document(rng: random.Random) -> str:
    kind = rng.choice(LEGAL_DOC_TYPES)
    return f"{kind} {rng.randint(1, 220)}/{rng.randint(2019, 2026)}/{rng.choice(LEGAL_DOC_AGENCIES)}"


def random_date_pair(rng: random.Random) -> tuple[str, str]:
    """2 giá trị DATE khác nhau, dùng cho seed liệt kê cùng loại (khắc phục lỗi
    'gộp cụm' 2026-09-15: model gộp '2h, 4h đến trên 6h' thành 1 entity vì DATE
    chỉ có 6/1000 ví dụ liệt kê liền nhau trong data V2 cũ, chủ yếu mượn khuôn
    LOCATION)."""
    kind = rng.choice(["hour", "year", "day"])
    if kind == "hour":
        h1, h2 = sorted(rng.sample(range(0, 24), 2))
        return f"{h1}h", f"{h2}h"
    if kind == "year":
        y1, y2 = sorted(rng.sample(range(2019, 2027), 2))
        return f"năm {y1}", f"năm {y2}"
    d1, d2 = sorted(rng.sample(range(1, 28), 2))
    m = rng.randint(1, 12)
    return f"ngày {d1} tháng {m}", f"ngày {d2} tháng {m}"


# Loại thực thể đủ pool/cơ chế để ép sinh 1 câu LIỆT KÊ LIỀN NHAU 2 giá trị cùng
# loại (phải tách thành 2 entity riêng, không gộp) — khắc phục đúng lỗ hổng đã
# đo được 2026-09-15: 87/112 case liệt-kê-cùng-loại trong V2 cũ chỉ là LOCATION
# (địa chỉ hành chính, mẫu câu công thức), còn STATE_BODY/DATE/ORGANIZATION/LAW
# gần như không có ví dụ nào -> model không tổng quát hoá được quy tắc tách.
ENUM_ELIGIBLE_TYPES = ["STATE_BODY", "COURT", "ORGANIZATION", "LAW", "LOCATION", "LEGAL_ROLE", "DATE"]

SYSTEM_EXTRA = """Bạn là chuyên gia xây dựng dữ liệu huấn luyện cho hệ thống trích xuất \
thực thể pháp lý tiếng Việt. Bạn viết ĐOẠN VĂN thực tế (không phải danh sách), tự nhiên \
như văn bản pháp lý/tin tức/báo cáo thật, rồi gán nhãn CHÍNH XÁC các thực thể xuất hiện.

YÊU CẦU BẮT BUỘC:
1. Đoạn văn phải TỰ NHIÊN đưa các thực thể được gợi ý (ở dưới) vào, đúng NGUYÊN VĂN \
chuỗi ký tự đã cho (không đổi cách viết, không viết tắt khác đi).
2. Có thể thêm các thực thể KHÁC không có trong gợi ý nếu đoạn văn cần (ví dụ ngày \
tháng, số tiền, điều luật cụ thể) — miễn đúng 12 loại và đúng quy tắc.
3. Nhãn "entities" phải liệt kê ĐẦY ĐỦ và CHÍNH XÁC mọi thực thể thực sự xuất hiện \
trong đoạn văn, kể cả các thực thể gợi ý.
4. Không lặp lại nguyên văn cấu trúc câu mở đầu giữa các đoạn (đa dạng cách diễn đạt).

Chỉ trả về JSON: {"paragraph": "...", "entities": [{"text": "...", "type": "..."}]}"""


def build_user_prompt(genre_desc: str, seeds: dict[str, str], enum_seed: tuple[str, str, str] | None = None) -> str:
    seed_lines = "\n".join(f"- {t}: {v}" for t, v in seeds.items())
    enum_block = ""
    if enum_seed is not None:
        etype, val_a, val_b = enum_seed
        enum_block = f"""

YÊU CẦU THÊM (bắt buộc): trong đoạn văn phải có 1 câu LIỆT KÊ LIỀN NHAU (nối bằng \
dấu phẩy hoặc "và") CẢ HAI giá trị sau, đều thuộc loại {etype}: "{val_a}" và "{val_b}". \
Khi gán nhãn, đây PHẢI là 2 entity {etype} TÁCH RIÊNG (mỗi entity chỉ chứa đúng 1 \
trong 2 chuỗi trên) — KHÔNG được gộp cả hai (kèm từ nối ở giữa) thành một entity duy nhất."""
    return f"""Viết MỘT đoạn văn tiếng Việt (100-220 từ), thể loại: {genre_desc}

Các thực thể GỢI Ý cần đưa vào đoạn văn (giữ nguyên văn cách viết):
{seed_lines}{enum_block}

Trả về đúng JSON theo format đã nêu."""


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=180)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data-finetune-v3-task1.1"))
    ap.add_argument("--system-prompt-source", type=Path,
                     default=Path("data-finetune-v2/data_finetune_v2_train.jsonl"),
                     help="Chỉ lấy NGUYÊN VĂN system prompt V2 hiện có (định dạng bài toán), "
                          "không đọc bất kỳ nội dung/nhãn nào khác từ file này.")
    args = ap.parse_args()

    load_dotenv()
    model = args.model or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6"
    client = anthropic.Anthropic()

    system_prompt = None
    with args.system_prompt_source.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("group") == "V2":
                system_prompt = row["messages"][0]["content"]
                break
    if system_prompt is None:
        raise SystemExit("Không tìm thấy mẫu V2 trong file system-prompt-source.")

    rng = random.Random(args.seed)
    total_n = args.n_train + args.n_dev + args.n_test

    state_body_pool = RoundRobinPool(build_state_body_pool(rng), rng)
    court_pool = RoundRobinPool(build_court_pool(), rng)
    org_pool = RoundRobinPool(build_org_pool(rng), rng)
    law_pool = RoundRobinPool(LAW_POOL, rng, max_uses=max(MAX_USES_PER_VALUE, total_n // len(LAW_POOL) + 5))
    location_pool = RoundRobinPool(LOCATION_POOL, rng)
    legal_role_pool = RoundRobinPool(LEGAL_ROLE_POOL, rng, max_uses=max(MAX_USES_PER_VALUE, total_n // len(LEGAL_ROLE_POOL) + 5))

    def make_spec() -> dict:
        genre_key, genre_desc = rng.choice(GENRES)
        seeds = {
            "PERSON": random_person(rng),
            "STATE_BODY": state_body_pool.next(),
            "COURT": court_pool.next(),
            "LAW": law_pool.next(),
            "LEGAL_DOCUMENT": random_legal_document(rng),
            "LOCATION": location_pool.next(),
            "LEGAL_ROLE": legal_role_pool.next(),
            "CASE_ID": random_case_id(rng),
        }
        # đoạn văn không nhất thiết cần MỌI loại -- rút ngẫu nhiên 4-6 gợi ý, tránh
        # công thức cứng nhắc lặp lại y hệt giữa các mẫu.
        keys = rng.sample(list(seeds), k=rng.randint(4, 6))
        if rng.random() < 0.35:
            keys.append("ORGANIZATION")
            seeds["ORGANIZATION"] = org_pool.next()
        chosen = {k: seeds[k] for k in keys}

        # ~30% cac mau: ep 1 cau liet ke lien nhau 2 gia tri CUNG LOAI (khac phuc
        # lo hong "gop cum" - xem ENUM_ELIGIBLE_TYPES o tren). Chia deu cac loai
        # de STATE_BODY/DATE/COURT/... duoc luyen tap nhu LOCATION, khong lech.
        enum_seed = None
        if rng.random() < 0.30:
            etype = rng.choice(ENUM_ELIGIBLE_TYPES)
            pool_map = {
                "STATE_BODY": state_body_pool, "COURT": court_pool, "ORGANIZATION": org_pool,
                "LAW": law_pool, "LOCATION": location_pool, "LEGAL_ROLE": legal_role_pool,
            }
            if etype == "DATE":
                val_a, val_b = random_date_pair(rng)
            else:
                val_a = pool_map[etype].next()
                val_b = pool_map[etype].next()
                if val_a == val_b:
                    val_b = pool_map[etype].next()
            if val_a != val_b:
                enum_seed = (etype, val_a, val_b)

        return {
            "genre_key": genre_key, "genre_desc": genre_desc, "seeds": chosen,
            "difficulty": rng.choice(DIFFICULTIES), "enum_seed": enum_seed,
        }

    lock = threading.Lock()
    collected: list[dict] = []
    seen_paragraphs: set[str] = set()
    stats = {"calls": 0, "api_err": 0, "parse_err": 0, "rejected_ungrounded": 0,
             "rejected_dup": 0, "rejected_badtype": 0, "rejected_enum_merged": 0,
             "in_tok": 0, "out_tok": 0}

    def run_one(spec: dict) -> None:
        if len(collected) >= total_n:
            return
        user = build_user_prompt(spec["genre_desc"], spec["seeds"], spec.get("enum_seed"))
        try:
            resp = client.messages.create(
                model=model, max_tokens=1500,
                system=system_prompt + "\n\n---\n\n" + SYSTEM_EXTRA,
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
        if not obj or not isinstance(obj.get("entities"), list) or not obj.get("paragraph"):
            with lock:
                stats["parse_err"] += 1
            return

        paragraph = norm(str(obj["paragraph"]).strip())
        raw_entities = obj["entities"]
        validated = []
        seen_pairs = set()
        ok = True
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            etype = str(e.get("type", "")).strip().upper()
            etext = norm(str(e.get("text", "")).strip())
            if not etext or etype not in ENTITY_TYPES:
                continue
            if etext not in paragraph:
                ok = False
                break
            key = (etype, etext.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            validated.append({"text": etext, "type": etype})
        if not ok:
            with lock:
                stats["rejected_ungrounded"] += 1
            return
        if len(validated) < 2:
            with lock:
                stats["rejected_badtype"] += 1
            return

        enum_seed = spec.get("enum_seed")
        if enum_seed is not None:
            etype, val_a, val_b = enum_seed
            keys_present = {k for k in seen_pairs if k[0] == etype}
            if (etype, val_a.lower()) not in keys_present or (etype, val_b.lower()) not in keys_present:
                # model gop chung hoac bo sot 1 trong 2 gia tri -> loai, khong dua
                # vao data (tranh day nguoc lai dung loi da tim thay)
                with lock:
                    stats["rejected_enum_merged"] += 1
                return

        dup_key = re.sub(r"\s+", " ", paragraph.lower())[:120]
        with lock:
            if dup_key in seen_paragraphs:
                stats["rejected_dup"] += 1
                return
            if len(collected) >= total_n:
                return
            seen_paragraphs.add(dup_key)
            collected.append({
                "paragraph": paragraph, "entities": validated,
                "genre": spec["genre_key"], "difficulty": spec["difficulty"],
            })

    n_calls = int(total_n * 1.6)  # dư bù mẫu bị loại (ungrounded/dup/parse err/enum bị gộp)
    specs = [make_spec() for _ in range(n_calls)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, s) for s in specs]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                print(f"  ... {len(collected)}/{total_n} mau  ({i}/{len(futures)} luot goi)", flush=True)

    rng.shuffle(collected)
    collected = collected[:total_n]
    if len(collected) < total_n:
        print(f"CANH BAO: chi thu duoc {len(collected)}/{total_n} mau dat yeu cau.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        ("train", collected[: args.n_train]),
        ("dev", collected[args.n_train: args.n_train + args.n_dev]),
        ("test", collected[args.n_train + args.n_dev: args.n_train + args.n_dev + args.n_test]),
    ]
    for split_name, items in splits:
        out_path = args.out_dir / f"task_1_1_v3_{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for idx, item in enumerate(items, 1):
                row = {
                    "id": f"V2G-{split_name[:2].upper()}-{idx:04d}",
                    "group": "V2",
                    "task_family": "benchmark",
                    "task_name": "legal_entity_extraction",
                    "benchmark_alignment": "VietLegal-1.1",
                    "slice": item["genre"],
                    "difficulty": item["difficulty"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["paragraph"]},
                        {"role": "assistant",
                         "content": json.dumps({"entities": item["entities"]}, ensure_ascii=False,
                                                separators=(",", ":"))},
                    ],
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(items)} mau -> {out_path}")

    # thong ke phan bo gia tri de tu kiem tra ngay (khong dua vao benchmark)
    def value_share_report(name: str, rr_pool: RoundRobinPool) -> None:
        used = {v: c for v, c in rr_pool.usage.items() if c > 0}
        if not used:
            return
        top = sorted(used.items(), key=lambda kv: -kv[1])[:3]
        total_used = sum(used.values())
        print(f"  {name}: {len(used)} gia tri khac nhau da dung, top3 = "
              f"{[(v, c, f'{c/total_used:.1%}') for v, c in top]}")

    print("\nPhan bo gia tri (rui ro cao) sau khi sinh:")
    value_share_report("STATE_BODY", state_body_pool)
    value_share_report("COURT", court_pool)
    value_share_report("ORGANIZATION", org_pool)
    value_share_report("LAW", law_pool)
    value_share_report("LOCATION", location_pool)
    value_share_report("LEGAL_ROLE", legal_role_pool)

    cost = stats["in_tok"] / 1e6 * 3 + stats["out_tok"] / 1e6 * 15
    print(f"\nDa sinh {len(collected)} mau hop le.")
    print(f"Luot goi API: {stats['calls']} (loi API {stats['api_err']}, loi parse {stats['parse_err']})")
    print(f"Loai bo: {stats['rejected_ungrounded']} khong grounding, "
          f"{stats['rejected_badtype']} qua it entity hop le, {stats['rejected_dup']} trung lap, "
          f"{stats['rejected_enum_merged']} enum-seed bi gop cum")
    print(f"Token: {stats['in_tok']:,} in / {stats['out_tok']:,} out  (~${cost:.2f} voi {model})")


if __name__ == "__main__":
    main()
