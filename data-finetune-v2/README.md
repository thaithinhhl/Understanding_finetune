# Thư mục `data-finetune-v2`

Dữ liệu huấn luyện/đánh giá dạng hội thoại (`messages`), dùng trực tiếp cho
`scripts/train_qlora.py`. Mỗi dòng JSONL có schema:

```json
{"id": "...", "group": "...", "task_family": "...", "task_name": "...",
 "benchmark_alignment": "...", "slice": "...", "difficulty": "...",
 "messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}
```

## Bảng nhóm task (`group`)

| Group | `task_family` | `task_name` | Nội dung |
|---|---|---|---|
| C1 | understanding | `question_rewrite_compatibility` | Viết lại câu hỏi thành câu độc lập |
| U1 | understanding | `intent_understanding` | Nhận diện ý định (9 nhãn) |
| U2 | understanding | `legal_concept_understanding` | Trích khái niệm pháp lý |
| U3 | understanding | `scope_user_facts_understanding` | Trích phạm vi tra cứu + tình tiết người dùng |
| V1 | benchmark | `legal_topic_classification` | Phân loại lĩnh vực (VietLegal-1.2) |
| V2 | benchmark | `legal_entity_extraction` | Trích xuất thực thể (VietLegal-1.1) |
| V3 | benchmark | `user_intent_understanding` | Phân loại ý định đa nhãn (VietLegal-2.5) |

C1/U1/U2/U3 dùng **chung một system prompt** (trả JSON 7 field); V1/V2/V3 mỗi
nhóm có system prompt riêng.

---

## Danh sách file

### Bộ chính (đa task, 7 nhóm)

| File | Số dòng | Thành phần | Dùng để |
|---|---|---|---|
| `data_finetune_v2_train.jsonl` | 7000 | 1000 mẫu × 7 nhóm | Train chính — đã dùng cho `adapters/7b/qwen2.5-7b-legal-v2` |
| `data_finetune_v2_dev.jsonl` | 700 | 100 × 7 nhóm | Validation trong lúc train |
| `data_finetune_v2_test.jsonl` | 1260 | 180 × 7 nhóm | Test held-out |

**Lưu ý quan trọng về file test:** đây **đã là bản đã khử trùng lặp**. Bản gốc có
42/1260 mẫu trùng khuôn mẫu với tập train (Jaccard ≥ 0.9, chủ yếu slice
`multi_scope` của U3: cùng tên văn bản luật, chỉ khác số Điều/Khoản). 42 mẫu đó đã
được sinh lại với nội dung mới (số hiệu/điều/khoản khác, hoặc đổi chủ đề), gold
JSON cập nhật tương ứng. Sau khi sửa: 0 mẫu còn Jaccard ≥ 0.9.

Điểm số trước/sau khi sửa gần như không đổi (xem `RESULTS.md` mục 1b) → kết quả cao
không đến từ việc model thuộc khuôn mẫu.

### File lọc theo nhóm

| File | Số dòng | Dùng để |
|---|---|---|
| `data_finetune_v2_test_v3.jsonl` | 180 | Lọc **chỉ nhóm V3** từ `data_finetune_v2_test.jsonl`. Cần thiết vì `scripts/eval_task_2_5_v3.py` chỉ hiểu schema gold của V3 (`{"intents": [...]}`); đưa cả 1260 dòng vào sẽ lỗi do các nhóm khác có schema khác. Không phải file mới, chỉ là tập con. |

### Bộ data task 2.5 sinh lại (v4)

Sinh bằng `scripts/generate_task_2_5_data_v4.py` (Claude API, `claude-sonnet-4-6`),
nhằm chữa bệnh đã chẩn đoán ở bộ V3 cũ: 91.6% câu 4 nhãn được ghép máy móc bằng
từ nối liệt kê (", đồng thời", ". Ngoài ra"…), khiến model chỉ học "đếm mệnh đề".

| File | Số dòng | Dùng để |
|---|---|---|
| `data_finetune_v2_v3_regen.jsonl` | 1000 | **Lô sinh gốc** (seed 42). Là nguyên liệu thô, chưa chia tập. Giữ lại để truy vết. |
| `data_2_5_v4_train.jsonl` | 1000 | Train — đã dùng cho `adapters/7b/qwen2.5-7b-task2.5-v4` |
| `data_2_5_v4_dev.jsonl` | 100 | Validation |
| `data_2_5_v4_test.jsonl` | 180 | Test held-out |

Ba file `data_2_5_v4_*` được chia ra từ 1399 mẫu = lô gốc (1000) + lô bổ sung
(399, seed 4242), đã khử trùng lặp chéo giữa hai lô.

**Nguyên tắc gán nhãn của bộ này:** mỗi nhãn ứng với một ý định **thật sự có** trong
câu hỏi. Câu quá mơ hồ → đúng một nhãn `general` (liên quan pháp luật) hoặc
`chitchat` (không liên quan). Đây **khác** với định nghĩa của benchmark
`vlegal_task_2_5_test.jsonl` (gán nhiều nhãn cho câu mơ hồ = "các hướng xử lý chưa
loại trừ được") — xem `RESULTS.md` mục 4c.

**Kiểm định chống overfit** (bằng `scripts/audit_generated_2_5.py`):

| Phép kiểm | Kết quả | Ngưỡng |
|---|---|---|
| Rò rỉ benchmark (Jaccard p95) | 0.28 | < 0.5 — đạt |
| Baseline bag-of-words Micro-F1 | 74.4% (cũ: 83.1%) | < 70% — **chưa đạt** |
| % dùng từ nối liệt kê máy móc | 0.0% (cũ: 91.6%) | đạt |

Script sinh **không đọc** file benchmark nào, kể cả làm ví dụ few-shot.

### Khác

| File | Ghi chú |
|---|---|
| `.DS_Store` | Rác do macOS tạo, không liên quan dữ liệu — có thể xoá |

---

## Phân biệt với thư mục khác

| Thư mục | Nội dung |
|---|---|
| `data-finetune-v2/` | Dữ liệu **train/dev/test** dạng hội thoại (thư mục này) |
| `data-benchmark-v2/` | Bộ **benchmark ngoài**, định dạng trắc nghiệm (`instruction`/`question`/`answers`/`ground_truth`) — task 1.1 và 1.2 |
| `vlegal_task_2_5_test.jsonl` (ở gốc repo) | Benchmark ngoài cho task 2.5, 1359 câu trắc nghiệm A-D đa nhãn |
| `data_train/` | Bộ dữ liệu **thế hệ trước**, giữ lại để đối chiếu |
