# Task 1.2 — Hiện trạng lỗi và hướng cải thiện dữ liệu

Chốt tại lần chạy mới nhất: model 7B fine-tune, bộ eval v2 (`data/vlegal_task_1_2_test_v2.jsonl`),
4-bit + system prompt ràng buộc + constrained decoding.

## 1. Hiện trạng

| Model (cùng cấu hình) | Accuracy |
|---|---|
| 7B chưa fine-tune | 562/683 = 82.28% |
| **7B fine-tune** | **587/683 = 85.94%** |

Quá trình cải thiện đã đi qua:

| Bước | Accuracy | Nguồn cải thiện |
|---|---|---|
| Prompt gốc (không ràng buộc), eval v1 | 77.01% | — |
| + System prompt "chỉ trả lời 1 chữ cái A-F" | 80.97% | +3.96 |
| + Constrained decoding (ép logit chỉ A-F) | 81.41% | +0.44 |
| + Eval v2 (sửa 40 nhãn sai của benchmark) | **85.94%** | +4.53 |

## 2. Lỗi đã xử lý xong — không còn tái diễn

**Lỗi định dạng: 0/683 case.** Trước đây model bịa đáp án "G" (49 case), trả lời bằng tên lĩnh vực
thay vì chữ cái (8 case). Constrained decoding chặn ở tầng logit nên về mặt toán học model không
thể sinh token ngoài A-F. Không cần làm gì thêm cho nhóm lỗi này.

## 3. Lỗi còn lại — 96 case, đều là lỗi nội dung

### 3.1. Thiên lệch theo vị trí đáp án đúng

| Vị trí | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| Accuracy | **78.0%** | 81.2% | 86.5% | 90.5% | 86.6% | **94.7%** |

Chênh gần **17 điểm** giữa vị trí A và F dù độ khó nội dung như nhau (thứ tự lựa chọn được xáo
trộn ngẫu nhiên trong bộ test). Model base (chưa fine-tune) cũng mắc đúng dạng này (A: 70%, F: 93%)
→ **đây là đặc tính của Qwen2.5 base, không phải do fine-tune gây ra, và không sửa được bằng prompt.**

### 3.2. Nhầm lẫn giữa các lĩnh vực gần nghĩa

| Lĩnh vực (ground truth) | Tỷ lệ sai |
|---|---|
| **Đầu tư** | 16/30 = 53.3% |
| **Lĩnh vực khác** | 10/21 = 47.6% |
| **Quyền dân sự** | 10/23 = 43.5% |
| Thủ tục tố tụng | 7/22 = 31.8% |
| Thương mại | 7/22 = 31.8% |
| Doanh nghiệp | 6/22 = 27.3% |

Các cặp nhầm lẫn lặp lại nhiều nhất:

| Đáp án đúng → Model chọn nhầm | Số lần |
|---|---|
| Đầu tư → Xây dựng - đô thị | 5 |
| Đầu tư → Dịch vụ pháp lý | 3 |
| Bảo hiểm → Lao động - tiền lương | 3 |
| Quyền dân sự → Bộ máy hành chính | 2 |
| Quyền dân sự → Dịch vụ pháp lý | 2 |
| Đầu tư → Doanh nghiệp | 2 |

Đặc điểm chung: đây là những lĩnh vực **thật sự chồng lấn ngữ nghĩa**. Câu hỏi về đấu thầu dự án
xây dựng vừa thuộc "Đầu tư" vừa liên quan "Xây dựng - đô thị"; câu hỏi về bảo hiểm xã hội cho người
lao động vừa là "Bảo hiểm" vừa là "Lao động - tiền lương". Model chưa học được ranh giới phân định.

## 4. Việc cần làm để cải thiện tiếp

Không còn gì "vá" được ở tầng inference. Hai hướng duy nhất, đều nằm ở dữ liệu train:

### 4.1. Bổ sung hard negative cho các cặp chồng lấn

Khi sinh dữ liệu train mới (theo `1.2_fix_data_traning.md`), với mỗi câu hỏi thuộc các lĩnh vực
hay bị nhầm, **bắt buộc đưa lĩnh vực hay nhầm vào làm 1 trong 5 lựa chọn nhiễu**, thay vì chọn
nhiễu ngẫu nhiên:

```
Đầu tư          -> luôn kèm nhiễu: Xây dựng - đô thị, Dịch vụ pháp lý, Doanh nghiệp
Bảo hiểm        -> luôn kèm nhiễu: Lao động - tiền lương
Quyền dân sự    -> luôn kèm nhiễu: Bộ máy hành chính, Dịch vụ pháp lý
Thủ tục tố tụng -> luôn kèm nhiễu: Dịch vụ pháp lý, Trách nhiệm hình sự
Thương mại      -> luôn kèm nhiễu: Dịch vụ pháp lý, Xuất nhập khẩu
```

Ưu tiên bổ sung số lượng câu hỏi cho 3 lĩnh vực sai nặng nhất: **Đầu tư, Lĩnh vực khác, Quyền dân sự**.

### 4.2. Cân bằng vị trí đáp án đúng khi sinh dữ liệu

Với mỗi câu hỏi gốc, sinh đủ 6 biến thể đặt đáp án đúng lần lượt ở A→F. Mục tiêu kéo chênh lệch
accuracy giữa vị trí A và F từ 17 điểm xuống dưới 5 điểm.

### 4.3. Xem lại định nghĩa nhãn "Lĩnh vực khác"

47.6% sai — nhãn này mơ hồ, dùng như thùng rác chứa mọi câu không phân loại được. Cần hoặc định
nghĩa rõ tiêu chí khi nào dùng, hoặc tách nhỏ thành các lĩnh vực cụ thể hơn.

## 5. File trong thư mục này

| File | Nội dung |
|---|---|
| `wrong_cases_v2.json` | 96 case còn sai: câu hỏi, 6 lựa chọn, đáp án đúng, model chọn gì |
| `confusion_pairs.json` | 74 cặp nhầm lẫn, sắp xếp theo số lần xuất hiện (17 cặp lặp ≥2 lần) |

Tài liệu liên quan:
- `1.2_fix_data_traning.md` — hướng dẫn chi tiết cách sinh lại dữ liệu train cho task 1.2
- `1.2_fix_ground_truth.json` — bản ghi review 127 case sai của bộ eval v1 (40 nhãn đã sửa)
- `predict/qwen2.5-7b-legal-lora-task-1.2-error-analysis.md` — phân tích lỗi chi tiết trên bộ eval v1
