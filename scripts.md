# Hướng dẫn chạy model 32B (4-bit) trên task Understanding

Áp dụng đúng 2 script sẵn có (`scripts/eval_understanding.py` + `scripts/score_understanding.py`) đã dùng cho model 7B, không cần viết code mới. Bộ test: `data_train/base_legal_7b_internal_test.jsonl`, 720 mẫu (4 task C1/U1/U2/U3, 180 mẫu/task).

## 1. Kiểm tra tài nguyên trước khi làm bất kỳ bước nào

| | Cần cho 32B | Hiện có trên máy này |
|---|---|---|
| Dung lượng đĩa để tải model | **~65 GB** (32B tham số × 2 byte, bf16 gốc từ HuggingFace) | **~4.9 GB trống** (`df -h /`) — **không đủ** |
| VRAM để chạy 4-bit inference | Ước tính **~18-20 GB** (32B × 0.5 byte cho 4-bit + overhead NF4 + KV cache) | GPU 24 GB — khả thi nhưng sát trần |

So sánh với model 7B đã chạy: 7B 4-bit chỉ cần ~5-6 GB VRAM và 15 GB đĩa. Model 32B nặng hơn ~4.3 lần.

**Việc bắt buộc phải làm trước tiên: dọn đủ ~65GB đĩa trống.** Cách khả thi nhất trên máy này:

```bash
# Kiểm tra dung lượng đang chiếm dụng
du -sh /root/VIET_LEGAL_TEAM/models/7b

# Nếu không cần chạy song song 7B và 32B, xoá tạm model 7B để lấy chỗ
# (LoRA adapter 7B ở adapters/7b/ vẫn giữ nguyên, chỉ mất base model,
#  cần tải lại bằng scripts/download_model.py khi muốn dùng lại)
rm -rf /root/VIET_LEGAL_TEAM/models/7b/Qwen2.5-7B-Instruct
```

Nếu không đủ chỗ ngay cả sau khi xoá 7B (15GB), cần mount thêm ổ đĩa ngoài trước khi tiếp tục — không có cách nào chạy 32B mà thiếu dung lượng tải model.

## 2. Tải model 32B base

```bash
python scripts/download_model.py Qwen/Qwen2.5-32B-Instruct models/32b/Qwen2.5-32B-Instruct
```

Kiểm tra tài nguyên sau khi tải xong (dùng script sẵn có của repo):

```bash
python scripts/check_environment.py --model-path models/32b/Qwen2.5-32B-Instruct
```

## 3. (Tuỳ chọn) LoRA adapter 32B

Nếu đã có adapter finetune riêng cho 32B — theo đúng đường dẫn mặc định trong `configs/qwen2.5-32b-qlora.yaml` là `adapters/32b/qwen2.5-32b-legal` — dùng `--adapter` để nạp lên trên base model. Nếu **chưa có** (trường hợp phổ biến, vì hiện tại chỉ mới finetune xong bản 7B), bỏ hẳn `--adapter` để chạy model 32B gốc — vẫn cho ra kết quả baseline hữu ích để so với 7B.

## 4. Chạy inference trên bộ test understanding

```bash
./legal/bin/python scripts/eval_understanding.py \
  --model models/32b/Qwen2.5-32B-Instruct \
  --adapter adapters/32b/qwen2.5-32b-legal \
  --data data_train/base_legal_7b_internal_test.jsonl \
  --output results/understanding/32b-predictions.json \
  --precision 4bit
```

Bỏ dòng `--adapter ...` nếu chưa có adapter 32B (xem mục 3).

**Ước tính thời gian**: model 7B chạy 720 mẫu (`max_new_tokens=384`) mất khoảng 1-1.5 giờ trên GPU này. Model 32B lớn hơn ~4.3 lần, nên có thể mất **4-6 giờ** — bắt buộc chạy nền (`run_in_background` nếu dùng qua Claude Code, hoặc `nohup ... &` nếu chạy tay) và theo dõi log định kỳ, không chờ trực tiếp.

Theo dõi VRAM trong lúc chạy để phát hiện sớm nguy cơ OOM:

```bash
watch -n 5 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 5. Chấm điểm (không cần GPU, chạy CPU vài giây)

```bash
./legal/bin/python scripts/score_understanding.py \
  --predictions results/understanding/32b-predictions.json \
  --data data_train/base_legal_7b_internal_test.jsonl \
  --output results/understanding/32b-metrics.json \
  --label "Qwen2.5-32B-Instruct (4bit)"
```

## 6. So sánh với kết quả 7B đã có

Thêm 1 dòng vào bảng trong `results/understanding/README.md` (đã có sẵn 2 dòng Chưa fine-tune / Fine-tune cho 7B), theo đúng format đang dùng: Accuracy, F1, Tỷ lệ không bịa cho từng task U1/U2/U3/C1.

## 7. Xử lý sự cố thường gặp

| Sự cố | Cách xử lý |
|---|---|
| OOM (hết VRAM) giữa chừng | Giảm `--max-new-tokens` (mặc định 384, có thể thử 256); đảm bảo không có process nào khác đang chiếm GPU (`nvidia-smi` trước khi chạy) |
| Hết đĩa giữa lúc tải model | Xoá `models/7b/Qwen2.5-7B-Instruct` (15GB) nếu không cần dùng song song, hoặc mount thêm ổ ngoài |
| Muốn kiểm tra nhanh trước khi chạy full 720 mẫu | Thêm `--limit 60` (15 mẫu/nhóm) để smoke-test, đúng tinh thần "smoke test rồi mới chạy chính thức" đã áp dụng cho pipeline train trong `README.md` |
| Tốc độ sinh quá chậm | Xác nhận `--attn-implementation sdpa` đang được dùng (mặc định của `eval_understanding.py`), không phải `eager` |
