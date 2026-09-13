#!/usr/bin/env python3
"""Supervised fine-tuning with 4-bit QLoRA for conversational Legal data."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from validate_data import to_messages


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("model", "data", "lora", "training"):
        if section not in config:
            raise ValueError(f"Config thiếu section: {section}")
    return config


def load_jsonl(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy data: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def render_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    default_system_prompt: str,
    num_proc: int,
) -> Dataset:
    original_columns = dataset.column_names

    def render(row: dict[str, Any]) -> dict[str, str]:
        messages = to_messages(row, default_system_prompt)
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    return dataset.map(render, remove_columns=original_columns, num_proc=num_proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume checkpoint mới nhất, hoặc truyền đường dẫn checkpoint.",
    )
    parser.add_argument("--max-steps", type=int, help="Override để smoke test")
    parser.add_argument("--output-dir", type=Path, help="Override thư mục checkpoint")
    parser.add_argument("--adapter-dir", type=Path, help="Override thư mục adapter cuối")
    args = parser.parse_args()
    cfg = read_config(args.config)
    model_cfg, data_cfg, lora_cfg, train_cfg = (
        cfg["model"], cfg["data"], cfg["lora"], cfg["training"]
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA yêu cầu CUDA GPU nhưng PyTorch không thấy CUDA.")
    if train_cfg.get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU/PyTorch không hỗ trợ BF16; đổi bf16=false và fp16=true.")

    set_seed(int(train_cfg.get("seed", 42)))
    model_path = resolve_path(model_cfg["name_or_path"])
    model_source = str(model_path) if model_path.exists() else model_cfg["name_or_path"]
    output_dir = args.output_dir or resolve_path(train_cfg["output_dir"])
    adapter_dir = args.adapter_dir or resolve_path(train_cfg["final_adapter_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_name = model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_name)
    use_4bit = bool(model_cfg.get("use_4bit", True))
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(
                model_cfg.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if use_4bit
        else None
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=quantization,
        device_map={"": local_rank},
        torch_dtype=compute_dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        )
    else:
        model.enable_input_require_grads()
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_data = render_dataset(
        load_jsonl(resolve_path(data_cfg["train_file"])),
        tokenizer,
        data_cfg.get("default_system_prompt", ""),
        int(data_cfg.get("num_proc", 1)),
    )
    validation_file = data_cfg.get("validation_file")
    eval_data = None
    if validation_file:
        eval_data = render_dataset(
            load_jsonl(resolve_path(validation_file)),
            tokenizer,
            data_cfg.get("default_system_prompt", ""),
            int(data_cfg.get("num_proc", 1)),
        )

    max_steps = args.max_steps if args.max_steps is not None else -1
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        max_steps=max_steps,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps" if eval_data is not None else "no",
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        load_best_model_at_end=eval_data is not None,
        metric_for_best_model="eval_loss" if eval_data is not None else None,
        greater_is_better=False if eval_data is not None else None,
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        tf32=bool(train_cfg.get("tf32", True)),
        seed=int(train_cfg.get("seed", 42)),
        report_to=train_cfg.get("report_to", "none"),
        dataset_text_field="text",
        max_seq_length=int(data_cfg.get("max_seq_length", 2048)),
        packing=bool(data_cfg.get("packing", False)),
        dataset_num_proc=int(data_cfg.get("num_proc", 1)),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    with (adapter_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, ensure_ascii=False, indent=2)
    print(f"Final adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
