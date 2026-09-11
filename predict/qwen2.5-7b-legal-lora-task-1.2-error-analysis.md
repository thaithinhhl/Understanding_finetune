# Phân tích lỗi — Task 1.2 (model fine-tune, không system prompt)

Nguồn dữ liệu: `predict/qwen2.5-7b-legal-lora-task-1.2.json` (683 mẫu) và
`predict/qwen2.5-7b-legal-lora-task-1.2-wrong-cases.csv` (157 case sai, đủ cả
câu hỏi/predict/ground_truth để tra cứu từng dòng).

**Accuracy: 526/683 = 77.01%** — 157 case sai, chia làm 5 loại lỗi rõ rệt.

## 1. Tổng quan 5 loại lỗi

| # | Loại lỗi | Số case | % tổng số mẫu | % trong 157 case sai |
|---|---|---|---|---|
| 1 | Bịa đáp án `"G"` trần trụi | 36 | 5.3% | 22.9% |
| 2 | Bịa `"G. <lĩnh vực tự nghĩ>"` | 13 | 1.9% | 8.3% |
| 3 | Trả lời nguyên văn tên lĩnh vực — **đúng nội dung, chỉ sai format** | 5 | 0.7% | 3.2% |
| 4 | Trả lời nguyên văn tên lĩnh vực — sai cả nội dung | 3 | 0.4% | 1.9% |
| 5 | Chọn 1 chữ cái hợp lệ (A-F) nhưng sai lĩnh vực | 100 | 14.6% | 63.7% |
| | **Tổng** | **157** | **23.0%** | **100%** |

Nhóm 1+2+3+4 = 57 case, đúng bằng số `invalid` (model không đưa ra được 1 chữ cái A-F hợp lệ). Nhóm 5 là 100 case còn lại — model có chọn 1 chữ cái đàng hoàng nhưng chọn nhầm nội dung.

## 2. Lỗi #1 và #2 — Bịa thêm đáp án "G" (49/157 = 31% tổng số lỗi)

Đề chỉ cho 6 lựa chọn A-F, nhưng model thường xuyên tự sinh thêm 1 lựa chọn thứ 7:

- **36 lần**: chỉ trả lời đúng 1 ký tự `"G"`.
- **13 lần**: bịa hẳn nội dung, ví dụ `"G. Phòng chống rửa tiền"`, `"G. Nghĩa vụ quân sự"`, `"G. Sinh trắc học"` — đều là những chủ đề **không có trong 6 lựa chọn của câu đó**.

**Nguyên nhân gốc**: đối chiếu với `data_train/base_legal_7b_train.jsonl`, task tương ứng (`V1 - legal_topic_classification`) được huấn luyện theo format hoàn toàn khác:

```json
{"topic": "Lao động - tiền lương"}
```

— chọn tự do trong **26 lĩnh vực cố định**, không có khái niệm "trắc nghiệm A-F". Model chưa từng học cách tự giới hạn vào 1 tập con nhỏ được liệt kê ngay trong câu hỏi. Khi câu trả lời "thật" của nó không khớp bất kỳ lựa chọn nào trong 6 cái được cho, nó không ép mình chọn bừa mà tự tạo thêm 1 slot "G" để nhét câu trả lời đó vào.

**Đối chứng với model gốc (chưa fine-tune)**: chỉ 3/48 case invalid của model gốc liên quan đến "G" (6%), còn lại là trả lời dài dòng tự nhiên. Ở model fine-tune, tỷ lệ này là 49/57 (86%). → Đây là hành vi phát sinh **do fine-tune**, không phải đặc tính vốn có của model.

**Nội dung bịa ra không hề ngẫu nhiên** — nhiều trường hợp còn hợp lý/chính xác hơn cả ground truth gốc:

| Câu hỏi (tóm tắt) | Ground truth | Model bịa |
|---|---|---|
| Thu thập dữ liệu sinh trắc học làm CCCD | Quyền dân sự | **Sinh trắc học** (sát nghĩa hơn) |
| Chuyển đổi số, cuộc thi CNTT | Lĩnh vực khác | **Công nghệ thông tin** (đúng hơn nhãn "khác") |
| Chuẩn bị nghĩa vụ quân sự | Quyền dân sự | **Nghĩa vụ quân sự** (đúng hơn) |

→ Model không "hỏng", nó vẫn đang cố phân loại đúng theo phản xạ đã học — chỉ là làm sai định dạng bài thi.

## 3. Lỗi #3 và #4 — Trả lời nguyên văn tên lĩnh vực thay vì chữ cái (8/157)

8 case model bỏ qua yêu cầu "chọn chữ cái", viết thẳng tên lĩnh vực:

| Index | Ground truth | Model trả lời | Đánh giá |
|---|---|---|---|
| 5 | A. Giao thông vận tải | "Giao thông vận tải" | **Đúng nội dung** |
| 99 | B. Bất động sản | "Bất động sản" | **Đúng nội dung** |
| 337 | A. Giao thông vận tải | "Giao thông vận tải" | **Đúng nội dung** |
| 344 | B. Bất động sản | "Bất động sản" | **Đúng nội dung** |
| 565 | B. Bất động sản | "Bất động sản" | **Đúng nội dung** |
| 273 | C. Lĩnh vực khác | "Giao thông vận tải" | Sai nội dung |
| 321 | A. Doanh nghiệp | "Giao thông vận tải" | Sai nội dung |
| 361 | C. Tiền tệ ngân hàng | "Giao thông vận tải" | Sai nội dung |

**5/8 case model trả lời đúng 100% nội dung**, chỉ bị chấm sai vì hàm `extract_answer()` chỉ nhận diện được câu trả lời dạng chữ cái, không so khớp được với tên lĩnh vực trong `answers`. Nếu tính 5 case này là đúng, accuracy thực chất ≥ 531/683 = **77.75%**.

## 4. Lỗi #5 — Chọn sai lĩnh vực dù trả lời đúng format (100/157 = 64% tổng số lỗi)

### 4.1. Thiên lệch theo vị trí chữ cái (position bias)

| Vị trí đáp án đúng | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| Accuracy | **56.4%** | 72.7% | 79.3% | **86.7%** | 84.5% | 83.7% |

Chênh lệch 30 điểm % giữa vị trí A và D, dù nội dung khó/dễ ngang nhau (thứ tự A-F được xáo trộn ngẫu nhiên mỗi câu). Model có xu hướng **né tránh chọn "A"**: chỉ chọn "A" 73/683 lần dù đáng lẽ phải là ~117 lần nếu công bằng.

Đối chiếu với model gốc (chưa fine-tune) cho thấy dạng thiên lệch này **có sẵn từ model gốc** (A=60.7%, D=85.0%), không phải lỗi riêng của fine-tune — nhưng lỗi "bịa G" ở mục 2 lại **tập trung đúng vào vị trí A/B** (30/54 case, 55%), nên khuếch đại thêm điểm yếu sẵn có này riêng ở bản fine-tune.

### 4.2. Lĩnh vực dễ bị chọn sai nhất

| Lĩnh vực (ground truth) | Tỷ lệ sai |
|---|---|
| **Đầu tư** | 19/27 = **70.4%** |
| **Quyền dân sự** | 16/24 = **66.7%** |
| **Thương mại** | 17/30 = **56.7%** |
| Doanh nghiệp | 10/23 = 43.5% |
| Tiền tệ ngân hàng | 12/28 = 42.9% |
| Thủ tục tố tụng | 9/23 = 39.1% |
| Bộ máy hành chính | 9/24 = 37.5% |
| Lĩnh vực khác | 8/23 = 34.8% |

### 4.3. Cặp nhầm lẫn phổ biến nhất (đúng → model chọn nhầm)

| Nhầm lẫn | Số lần |
|---|---|
| Đầu tư → Xây dựng - đô thị | 5 |
| Đầu tư → Dịch vụ pháp lý | 3 |
| Bảo hiểm → Lao động - tiền lương | 3 |
| Thương mại → Dịch vụ pháp lý | 3 |
| Quyền dân sự → Bộ máy hành chính | 2 |
| Quyền dân sự → Lĩnh vực khác | 2 |
| Doanh nghiệp → Đầu tư | 2 |

Đa số là các lĩnh vực có ranh giới ngữ nghĩa gần nhau (Đầu tư/Xây dựng-đô thị, Bảo hiểm/Lao động, Thương mại/Dịch vụ pháp lý) — lỗi hiểu nhầm ngữ nghĩa thật, không phải lỗi ngẫu nhiên.

## 5. Đã thử sửa — kết quả thực nghiệm

| Cấu hình | Accuracy | Invalid |
|---|---|---|
| Baseline (không system prompt) | 77.01% | 57 |
| + Thêm system prompt ràng buộc | 80.97% | 13 |
| + Constrained decoding (ép logit chỉ A-F) | **81.41%** | **0** |

Constrained decoding xoá sạch invalid nhưng chỉ tăng thêm 0.44 điểm — vì khi bị ép chọn trong 6 lựa chọn, phần lớn case "muốn nói G" lại chọn nhầm sang 1 chữ khác (chỉ 3/13 case ép-chọn ra đúng). **Trần nâng cấp bằng cách sửa prompt/decoding chỉ dừng ở ~81%** — muốn vượt qua ngưỡng này phải sửa tận gốc dữ liệu train (thêm dữ liệu format trắc nghiệm A-F vào tập finetune), không còn "vá" được ở tầng inference nữa.

## 6. Tóm tắt khuyến nghị

| Vấn đề | Có sửa được ở tầng inference không? | Cách sửa |
|---|---|---|
| Bịa "G" (49 case) | Có, gần hết | System prompt + constrained decoding → còn ~13 case chọn sai nhưng không còn invalid |
| Trả lời bằng tên lĩnh vực thay vì chữ cái (8 case) | Có thể vá phần chấm điểm | Sửa `extract_answer()` để so khớp thêm với tên lĩnh vực trong `answers` |
| Thiên lệch vị trí A/B (đè lên toàn bộ 100 case wrong-valid) | Không | Cần retrain với dữ liệu cân bằng vị trí đáp án đúng |
| Nhầm lẫn ngữ nghĩa Đầu tư/Xây dựng-đô thị... | Không | Cần thêm dữ liệu train phân biệt rõ ranh giới các lĩnh vực gần nhau |

## File liên quan

- `predict/qwen2.5-7b-legal-lora-task-1.2.json` — kết quả đầy đủ 683 mẫu
- `predict/qwen2.5-7b-legal-lora-task-1.2-wrong-cases.csv` — 157 case sai, đủ câu hỏi/predict/ground_truth
- `predict/qwen2.5-7b-legal-lora-task-1.2-with-system-prompt.json` — thực nghiệm #1
- `predict/qwen2.5-7b-legal-lora-task-1.2-constrained.json` — thực nghiệm #2
