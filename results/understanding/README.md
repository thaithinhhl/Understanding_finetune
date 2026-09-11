# Kết quả đánh giá task Understanding

Bộ test: `data_train/base_legal_7b_internal_test.jsonl`, 720 mẫu (4 task × 180 mẫu).
Cấu hình: 4-bit, greedy decoding, prompt lấy nguyên văn từ file test.

## Kết quả chính

Mỗi task có 2 dòng — dòng "Chưa fine-tune" và dòng "Fine-tune" — để so trực tiếp model nào tốt hơn ở từng cột.

| Task | Việc model phải làm | Model | Accuracy | F1 (tính điểm một phần) | Tỷ lệ không bịa khi nên để trống |
|---|---|---|---|---|---|
| **U1** | Nhận diện ý định câu hỏi | 7B chưa fine-tune | 76.7% | 65.9% | – |
| | | 32B chưa fine-tune (AWQ 4bit) | 82.8% | 71.8% | – |
| | | **7B fine-tune** | **88.9%** | **86.1%** | – |
| **U2** | Trích khái niệm pháp lý | 7B chưa fine-tune | 18.9% | 17.7% | 35.3% |
| | | 32B chưa fine-tune (AWQ 4bit) | 29.4% | 27.0% | 35.3% |
| | | **7B fine-tune** | **62.2%** | **58.7%** | **94.1%** |
| **U3** | Trích phạm vi & tình tiết | 7B chưa fine-tune | 35.6% | 12.4% | 95.2% |
| | | 32B chưa fine-tune (AWQ 4bit) | 47.2% | 66.3% | 80.6% |
| | | **7B fine-tune** | **84.4%** | **94.7%** | **100.0%** |
| **C1** | Viết lại câu hỏi cho độc lập | 7B chưa fine-tune | 35.6% | – | – |
| | | 32B chưa fine-tune (AWQ 4bit) | 41.1% | – | – |
| | | **7B fine-tune** | **62.2%** | – | – |

Fine-tune cải thiện rõ rệt ở **cả 4 task**, khác hẳn với 3 benchmark trắc nghiệm A-F (task 1.2/2.4/2.5) — lý do là 4 task understanding này (C1/U1/U2/U3) dùng đúng định dạng JSON mà model đã được huấn luyện, còn benchmark A-F dùng định dạng model chưa từng thấy lúc train.

**Phát hiện quan trọng: model lớn hơn không thay thế được fine-tune.** 32B gốc (chưa fine-tune, chạy AWQ 4-bit) nhỉnh hơn 7B gốc ở mọi task nhờ quy mô lớn hơn — nhưng **vẫn thua xa 7B đã fine-tune** ở tất cả các chỉ số chính, có task cách biệt hơn 30 điểm % (U2: 29.4% vs 62.2%). Riêng `user_facts_micro` của 32B rơi về **0.0%** (TP=0, hoàn toàn không trích đúng tình tiết nào) — tệ hơn cả 7B chưa fine-tune. Kết luận: với domain hẹp như pháp lý Việt Nam, fine-tune đúng dữ liệu quan trọng hơn việc dùng model to hơn 4.5 lần.

### Accuracy nghĩa là gì ở mỗi task

- **U1**: % câu chọn đúng ý định (1 trong 9 nhãn có sẵn).
- **U2**: % câu trích **đúng toàn bộ** danh sách khái niệm pháp lý — thiếu 1 hoặc thừa 1 khái niệm cũng bị tính sai.
- **U3**: % câu trả lời **đúng cả 3 phần cùng lúc** — phạm vi tra cứu (điều/khoản/số văn bản), tình tiết người dùng, và có/không có tiền đề pháp lý.
- **C1**: % câu viết lại **khớp chữ với chữ** so với đáp án mẫu.

### F1 nghĩa là gì

U2 và U3 đòi model trả về một **danh sách** (khái niệm, phạm vi...). Accuracy chỉ tính đúng/sai toàn bộ danh sách, nên trích đúng 2/3 khái niệm vẫn bị coi là sai hoàn toàn. F1 sửa điều đó: cho điểm theo tỷ lệ trích đúng bao nhiêu trong số đã trích ra, và bắt được bao nhiêu trong số cần trích. U1 không phải bài toán danh sách nhưng vẫn tính F1 vì lý do khác: một số ý định (nhãn) rất hiếm gặp, Accuracy dễ bị các nhãn phổ biến che khuất, F1 giúp lộ ra nhãn nào model chưa nhận biết được (xem mục dưới).

### "Tỷ lệ không bịa khi nên để trống" nghĩa là gì

Nhiều câu hỏi thực ra **không có** khái niệm/phạm vi nào cần trích (đáp án đúng là danh sách rỗng). Cột này đo: trong số những câu đáng lẽ phải để trống, model có đủ tỉnh táo để trả về rỗng hay lại cố bịa ra thứ gì đó không có trong câu hỏi.

## Ba điều đáng chú ý

### 1. Con số 35.6% của model gốc ở U3 là ảo — nhưng fine-tune thì thật

62/180 câu của U3 vốn dĩ đáp án đúng là "để trống hết", model chỉ cần im lặng là ăn điểm. Tách hai nhóm câu hỏi ra:

| Nhóm câu hỏi | Chưa fine-tune | Fine-tune |
|---|---|---|
| Câu phải trích ra thông tin thật (118 câu) | **4.2%** | **76.3%** |
| Câu đáng lẽ để trống (62 câu) | 95.2% | 100.0% |

Model gốc gần như **không trích xuất được gì** — điểm 35.6% chỉ đến từ việc im lặng đúng lúc. Sau fine-tune, model trích đúng thật (76.3%), không còn ăn may.

### 2. C1: model gốc càng viết lại càng sai, fine-tune sửa được phần lớn

Model gốc chỉ đúng ở những câu vốn đã đầy đủ, không cần sửa gì. Đến những câu thực sự cần viết lại (mục đích chính của task này) thì gần như thất bại — fine-tune cải thiện rõ nhưng câu rút gọn (`anaphora`) vẫn còn yếu:

| Loại câu hỏi | Chưa fine-tune | Fine-tune |
|---|---|---|
| Câu đã đầy đủ, không cần viết lại | 73.2% | 100.0% |
| Câu rút gọn, phải suy chủ ngữ từ câu trước | **11.3%** | **21.0%** |
| Câu đổi chủ đề giữa chừng | **0.0%** | **10.5%** |

### 3. U1: model gốc mù 2 loại câu hỏi quan trọng, fine-tune cải thiện nhưng chưa hết

Accuracy 76.7% của model gốc nhìn ổn, nhưng tách theo từng loại ý định thì lộ ra model gần như không nhận ra được 2 loại quan trọng nhất về an toàn — nếu nhận nhầm thành câu hỏi bình thường, hệ thống sẽ trả lời bừa thay vì hỏi lại hoặc từ chối:

| Loại câu hỏi | Chưa fine-tune | Fine-tune |
|---|---|---|
| Câu hỏi mơ hồ, không rõ ý | **0.0%** | 85.7% |
| Câu hỏi ngoài phạm vi pháp luật | **11.1%** | **21.1%** |

Fine-tune sửa gần như hoàn toàn việc nhận diện câu hỏi mơ hồ, nhưng vẫn còn yếu ở việc nhận ra câu hỏi ngoài phạm vi pháp luật — 21.1% vẫn là mức thấp cho một tiêu chí an toàn.

## Định dạng đầu ra

Model phải trả JSON đúng 7 field quy định.

| Task | Chưa fine-tune | Fine-tune |
|---|---|---|
| C1 | 100.0% | 100.0% |
| U1 | 100.0% | 100.0% |
| U2 | 100.0% | 100.0% |
| U3 | 99.4% | 98.3% |

Cả 2 model đều tuân thủ định dạng tốt (98–100%), nên việc tuân thủ định dạng không phải vấn đề — chênh lệch kết quả giữa 2 model hoàn toàn đến từ nội dung trả lời, không phải do lỗi format.

## File trong thư mục này

| File | Nội dung |
|---|---|
| `base-predictions.json` | Output thô của model chưa fine-tune |
| `base-metrics.json` | Toàn bộ số liệu chi tiết |
| `base-inference.log` | Log chạy inference |
| `finetune-*` | Tương tự, cho model đã fine-tune |

Định nghĩa đầy đủ, công thức tính từng metric: xem `metrics_understanding.md` ở thư mục gốc.

Cách chạy lại:

```bash
# Bước 1: sinh prediction (cần GPU, ~30 phút)
./legal/bin/python scripts/eval_understanding.py \
  --model models/7b/Qwen2.5-7B-Instruct \
  --adapter adapters/7b/qwen2.5-7b-legal \
  --data data_train/base_legal_7b_internal_test.jsonl \
  --output results/understanding/finetune-predictions.json \
  --precision 4bit

# Bước 2: chấm điểm (chỉ cần CPU, vài giây)
./legal/bin/python scripts/score_understanding.py \
  --predictions results/understanding/finetune-predictions.json \
  --data data_train/base_legal_7b_internal_test.jsonl \
  --output results/understanding/finetune-metrics.json
```

## Còn thiếu

Task C1 còn một cách chấm khác là xét câu viết lại có **đúng nghĩa** hay không, thay vì đòi khớp từng chữ như hiện tại. Cách này cần một model mạnh hơn đứng ra làm giám khảo, mà máy hiện chỉ có Qwen2.5-7B nên chưa làm được. Cần API key của một model ngoài, hoặc chấm tay 180 mẫu.
