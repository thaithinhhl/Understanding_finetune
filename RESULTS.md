# Tổng hợp kết quả đánh giá

Toàn bộ số liệu dưới đây đọc trực tiếp từ file kết quả trong `predict/` và
`results/`. Mỗi bảng ghi rõ bộ test, cấu hình chạy và file nguồn.

Hai adapter được nhắc đến:

| Adapter | Train trên | Ghi chú |
|---|---|---|
| `adapters/7b/qwen2.5-7b-legal-v2` | `data-finetune-v2` đầy đủ (7000 mẫu, 7 nhóm task) | 3 epoch, QLoRA 4-bit |
| `adapters/7b/qwen2.5-7b-task2.5-v4` | chỉ data 2.5 sinh mới (1000 mẫu) | 3 epoch, thí nghiệm riêng cho task 2.5 |

---

## 1. Bốn task Understanding (C1/U1/U2/U3)

Bộ test: `data-finetune-v2/data_finetune_v2_test.jsonl`, 720 mẫu held-out
(4 task × 180). Chấm theo `metrics_understanding.md`. Chạy bf16 qua vLLM.

| Task | Metric chính | 7B base | 32B base | **7B fine-tune** |
|---|---|---|---|---|
| U1 – nhận diện ý định | Accuracy | 73.3% | 83.9% | **97.2%** |
| | Macro-F1 | 63.1% | 72.8% | **97.5%** |
| U2 – trích khái niệm | Exact Match | 18.3% | 27.8% | **74.4%** |
| | Restraint Rate | 35.3% | 35.3% | **100.0%** |
| U3 – trích scope/tình tiết | Structured EM | 36.7% | 50.0% | **86.1%** |
| | EM nhóm trích xuất | 11.0% | 30.5% | **78.8%** |
| C1 – viết lại câu hỏi | Exact Match | 38.3% | 42.2% | **70.0%** |
| Chung | JSON Valid Rate | 100% | 100% | 99.6% |

Nguồn: `results/understanding-v2/{base,32b,finetune-v2}-metrics-vllm.json`

**Kết luận:** fine-tune 7B vượt xa cả 32B chưa fine-tune ở cả 4 task — với domain
hẹp, fine-tune đúng dữ liệu quan trọng hơn tăng kích thước model 4.5 lần.

### 1b. Kiểm chứng chống rò rỉ dữ liệu

Phát hiện 42/1260 mẫu test trùng khuôn mẫu với train (chủ yếu slice `multi_scope`
của U3: cùng tên văn bản, chỉ khác số Điều/Khoản). Đã sinh lại 42 mẫu đó với nội
dung mới → `data_finetune_v2_test_dedup.jsonl` (0 mẫu còn Jaccard ≥ 0.9).

| Task | Test gốc | Test đã sửa | Chênh |
|---|---|---|---|
| U1 | 97.2% | 97.2% | 0 |
| U2 | 74.4% | 74.4% | 0 |
| U3 | 86.1% | 86.7% | +0.6 |
| C1 | 70.0% | 71.1% | +1.1 |

Điểm gần như không đổi → kết quả cao **không** đến từ việc thuộc khuôn mẫu.

Nguồn: `results/understanding-v2/finetune-v2-metrics-dedup-bf16.json`

---

## 2. Task 1.1 — Trích xuất thực thể pháp lý

Bộ test: `data-benchmark-v2/vlegal_task_1_1_test_v2_gold.jsonl`, 748 câu trắc
nghiệm 4 lựa chọn (A-D).

| Model | Cách chạy | Accuracy | Nguồn |
|---|---|---|---|
| 7B base | trắc nghiệm A-D, 4-bit | 69.92% | `predict/qwen2.5-7b-task-1.1.json` |
| **7B fine-tune** | trắc nghiệm A-D, 4-bit | **71.79%** | `predict/qwen2.5-7b-finetuned-v2-task-1.1.json` |
| 32B base | trắc nghiệm A-D, 4-bit | 91.18% | `predict/qwen2.5-32b-task-1.1.json` |
| 72B AWQ | trắc nghiệm A-D, 4-bit | **91.98%** | `predict/qwen2.5-72b-awq-task-1.1.json` |
| 7B fine-tune | prompt native (tự sinh JSON thực thể) | 35.16% | `predict/task-1.1-native-finetune.json` |

**Phân tích lỗi** (3438 thực thể của đáp án đúng, chế độ native):

| Kiểu lỗi | Tỷ lệ |
|---|---|
| Khớp hoàn toàn (type + text) | 41.0% |
| Bỏ sót hoàn toàn | 27.0% |
| Bị gộp vào chuỗi dài hơn | 19.3% |
| Bị cắt thành chuỗi ngắn hơn | 8.3% |
| Đúng text nhưng sai type | 4.3% |

Chỉ 1.93/4.20 thực thể model sinh ra nằm trong vùng 4 lựa chọn của đề → 54% là
trích ngoài phạm vi. Kết luận: **71.79% ở chế độ A-D không phản ánh năng lực trích
xuất thật** — nó chỉ đo khả năng nhận ra phương án đúng, dễ hơn nhiều so với tự
sinh chính xác ranh giới và nhãn thực thể.

---

## 3. Task 1.2 — Phân loại lĩnh vực pháp luật

Bộ test: `data-benchmark-v2/vlegal_task_1_2_test_v2.jsonl`, 683 câu, 6 lựa chọn A-F.

| Model / cách chạy | Accuracy |
|---|---|
| 7B base (constrained decoding + system prompt) | 82.28% |
| Adapter cũ `qwen2.5-7b-legal` | **85.94%** |
| **Adapter mới `qwen2.5-7b-legal-v2`** | **78.62%** |
| Adapter mới + prompt nhấn mạnh trắc nghiệm | 78.62% (đổi 50/683 dự đoán, tổng hoà bằng nhau) |
| Adapter mới + map ngược tự do (sinh tên lĩnh vực) | 49.49% (44% không map được) |
| Adapter mới + chấm log-likelihood trong 6 lựa chọn | 71.60% |

**Regression đã xác định nguyên nhân:** `data-finetune-v2` chỉ có 1000 mẫu V1 ở
định dạng JSON phân loại mở (`{"topic": "..."}`), **không có** biến thể trắc nghiệm
A-F. Model học trả JSON nên khả năng chọn 1 chữ cái suy giảm so với base.
Thiên lệch vị trí đáp án cũng bị khuếch đại: A 58.5% → F 90.4% (chênh 32 điểm,
base chỉ chênh 23 điểm).

Đổi prompt **không** sửa được (cả 3 cách đều ≤ 78.62%) vì lỗi nằm ở trọng số model,
không ở cách hỏi.

---

## 4. Task 2.5 — Phân loại ý định người dùng (đa nhãn)

### 4a. Trên benchmark thật (`vlegal_task_2_5_test.jsonl`, 1359 câu, A-D đa nhãn)

| Model | Prompt | Exact Match | Micro-F1 |
|---|---|---|---|
| 7B base | A-D + nhắc "chọn TẤT CẢ" | **17.00%** | **63.2%** |
| 7B fine-tune (legal-v2) | A-D + nhắc "chọn TẤT CẢ" | 15.16% | 62.5% |
| 7B fine-tune (legal-v2) | native JSON (như lúc train) | 16.63% | 42.1% |
| 7B fine-tune (legal-v2) | native JSON + nhắc nhiều intent | 14.42% | 44.5% |
| 7B fine-tune (task2.5-v4) | native JSON | 9.49% | 37.8% |

### 4b. Trên tập held-out cùng định nghĩa nhãn (180 mẫu)

| Model | Bộ test | Exact Match | Micro-F1 | avg pred / gold |
|---|---|---|---|---|
| 7B base | test V3 của `data-finetune-v2` | 27.78% | 68.4% | 1.86 / 2.50 |
| 32B base | test V3 của `data-finetune-v2` | 41.11% | 76.0% | 1.93 / 2.50 |
| 72B AWQ | test V3 của `data-finetune-v2` | 52.22% | 81.4% | 2.14 / 2.50 |
| **7B fine-tune (legal-v2)** | test V3 của `data-finetune-v2` | **96.11%** | **99.2%** | 2.47 / 2.50 |
| **7B fine-tune (task2.5-v4)** | test của data 2.5 sinh mới | **50.56%** | **82.1%** | 2.08 / 1.96 |

### 4c. Nguyên nhân khoảng cách giữa 4a và 4b

Hai bên **định nghĩa nhãn khác nhau về bản chất**:

| | Data train | Benchmark |
|---|---|---|
| Nhiều nhãn khi | câu chứa nhiều yêu cầu tường minh | câu **mơ hồ**, chưa loại trừ được hướng nào |
| `"Helo"` | `chitchat` (1 nhãn) | `[chitchat, legal_query, general, document_retrieval]` |

Đo được trên data v3 cũ: 91.6% câu 4 nhãn dùng từ nối liệt kê máy móc
(", đồng thời", ". Ngoài ra"), độ dài tăng đều +9 từ mỗi intent → model học
"đếm mệnh đề". Benchmark thì ngược lại: gần như 0% từ nối, và **câu càng ngắn
càng nhiều nhãn** (1-10 từ → 2.37 nhãn/câu; >60 từ → 1.79 nhãn/câu).

### 4d. Data sinh lại (v4) đã sửa được gì

| Chỉ số | data v3 cũ | **data v4 mới** |
|---|---|---|
| % câu 4 nhãn dùng từ nối máy móc | 91.6% | **0.0%** |
| % câu 3 nhãn dùng từ nối máy móc | 92.4% | **0.0%** |
| Baseline bag-of-words (Naive Bayes) Micro-F1 | 83.1% | **74.4%** (ngưỡng đạt < 70%) |
| Rò rỉ với benchmark (Jaccard p95) | — | **0.28** (đạt) |
| Model train xong: avg nhãn dự đoán / gold | 1.16 / 2.19 | **2.08 / 1.96** |

Data mới **đã chữa đúng bệnh đã chẩn đoán** (model hết dè dặt, dự đoán đủ số nhãn),
nhưng điểm benchmark vẫn giảm vì giữ nguyên định nghĩa nhãn cũ — đúng như dự báo
trước khi sinh. Hai chỉ số còn chưa đạt: baseline từ khoá 74.4% (cần < 70%) và độ
dài câu vẫn tương quan thuận với số nhãn (21.8 → 48.1 từ).

---

## 5. Task 2.4 — Đúng/Sai bản án

| Model | Accuracy | Nguồn |
|---|---|---|
| Adapter cũ `qwen2.5-7b-legal` | 84.47% (599 mẫu) | `predict/qwen2.5-7b-legal-lora-task-2.4.json` |

Bộ dữ liệu gốc của task này không có trên máy hiện tại (chỉ còn file kết quả cũ).

---

## 6. Ghi chú phương pháp

- **Constrained decoding** (task 1.1/1.2): ép logit chỉ sinh được 1 token trong tập
  chữ cái hợp lệ → 0% lỗi định dạng, nhưng chỉ đo khả năng *chọn phương án*.
- **Prompt native** (task 1.1/2.5): dùng đúng system prompt lúc train, để model tự
  sinh JSON, rồi map ngược về đáp án. Đo *năng lực tự sinh*, khó hơn nhiều.
- Chênh lệch lớn giữa hai cách (task 1.1: 71.79% vs 35.16%) cho thấy điểm số
  benchmark trắc nghiệm **không** thay thế được đánh giá năng lực sinh thật.
- vLLM chỉ chạy được bf16 (không hỗ trợ bnb 4-bit); các con số 4-bit đều chạy qua
  HF Transformers.
