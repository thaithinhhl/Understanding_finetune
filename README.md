- đây là repo có data1.2 new, đã sửa benchmark,
- có report file task understanding để chạy code trên các model 32B và 72B


# Vietnamese Legal LLM fine-tuning

Bộ khung SFT bằng QLoRA 4-bit cho Qwen2.5 7B/32B. Model, data, checkpoint và
adapter được lưu trên host; không được đóng vào Docker image.

## 1. Chuẩn bị GPU và môi trường

Tạm dừng Speech API khi đã xác nhận không có người dùng:

```bash
docker stop --time 30 viet-speech-app-1
nvidia-smi
```

Tạo môi trường. Cài PyTorch CUDA phù hợp trước theo hướng dẫn chính thức của
PyTorch, sau đó:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --workspace .
```

## 2. Tải model

```bash
python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct \
  models/7b/Qwen2.5-7B-Instruct
```

Kiểm tra model và tài nguyên:

```bash
python scripts/check_environment.py \
  --model-path models/7b/Qwen2.5-7B-Instruct
```

## 3. Data contract

Mỗi dòng JSONL dùng một trong hai schema:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

hoặc:

```json
{"instruction":"...","input":"...","output":"..."}
```

Đặt dữ liệu bàn giao tại `data/train.jsonl` và `data/validation.jsonl`. Script
chỉ kiểm tra, không sửa nội dung:

```bash
python scripts/validate_data.py data/train.jsonl data/validation.jsonl \
  --model models/7b/Qwen2.5-7B-Instruct --max-length 2048
```

## 4. Smoke test rồi train 7B

Smoke test 10 bước để kiểm tra VRAM, data, checkpoint và loss:

```bash
python scripts/train_qlora.py \
  --config configs/qwen2.5-7b-qlora.yaml --max-steps 10 \
  --output-dir checkpoints/smoke-7b \
  --adapter-dir adapters/smoke-7b
```

Nếu thành công, chạy chính thức bằng thư mục output đã khai báo trong config:

```bash
python scripts/train_qlora.py --config configs/qwen2.5-7b-qlora.yaml
```

Tiếp tục một run bị gián đoạn:

```bash
python scripts/train_qlora.py \
  --config configs/qwen2.5-7b-qlora.yaml --resume
```

Adapter cuối cùng được lưu tại `adapters/7b/qwen2.5-7b-legal`.

## 5. Test adapter

```bash
python scripts/infer_adapter.py \
  --adapter adapters/7b/qwen2.5-7b-legal \
  --prompt "Điều kiện có hiệu lực của giao dịch dân sự là gì?"
```

## 6. Model 32B

Cấu hình nằm tại `configs/qwen2.5-32b-qlora.yaml`. Không khuyến nghị chạy
pipeline 32B này trên một RTX 4090 24 GB. Script hiện tại yêu cầu toàn bộ model
4-bit nằm trên một GPU, vì vậy hãy dùng GPU >=48 GB. Muốn chia model qua nhiều
GPU cần bổ sung pipeline FSDP/DeepSpeed riêng; chỉ chạy nhiều process DDP không
giúp model vừa vào các GPU 24 GB. Luôn smoke test trước khi chạy toàn bộ.

## Ghi chú quan trọng

- Pipeline hiện huấn luyện loss trên toàn bộ chuỗi hội thoại, gồm system/user và
  assistant. Chỉ chuyển sang assistant-only loss sau khi đã kiểm thử chat
  template có generation mask đúng.
- `packing` mặc định tắt để các hội thoại pháp lý không bị ghép khó kiểm soát.
- Không dùng `device_map="auto"` để train; script gán một model vào GPU của
  process hiện tại.
- Sau khi train xong có thể bật lại Speech API bằng
  `docker start viet-speech-app-1`.
