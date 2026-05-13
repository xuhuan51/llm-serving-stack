#!/usr/bin/env python3
"""
LoRA SFT pipeline for Qwen2.5-7B / 14B / 32B on 8×A30 PCIe-only.

用 HuggingFace TRL + PEFT，支持:
  - 7B: DDP (单卡能装下，多卡纯加速)
  - 14B: ZeRO-2 (单卡装不下完整 14B，需切 optimizer + grad)
  - 32B: ZeRO-3 (weights 也要切到多卡)

启动方式:
  # smoke test (7B 单卡)
  accelerate launch --num_processes 1 train_lora_sft.py --model_path /home/liuguangli/models/Qwen2.5-7B-Instruct --max_steps 100

  # 8 卡 DDP (7B)
  accelerate launch --num_processes 8 train_lora_sft.py --model_path /home/liuguangli/models/Qwen2.5-7B-Instruct

  # 8 卡 ZeRO-3 (32B)
  deepspeed --num_gpus 8 train_lora_sft.py --model_path /home/liuguangli/models/Qwen2.5-32B-Instruct --deepspeed ds_zero3.json
"""
import argparse
import os
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from transformers.integrations import HfDeepSpeedConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="本地模型路径")
    p.add_argument("--dataset_path", default="/home/liuguangli/learn-ai-infra/training/data/alpaca_zh_5000.jsonl",
                   help="JSONL 数据集，字段 'instruction' / 'input' / 'output'")
    p.add_argument("--output_dir", default="/home/liuguangli/learn-ai-infra/training/runs/default")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 用于 smoke test")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--deepspeed", default=None, help="ds_zero2.json or ds_zero3.json")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="省 activation 显存的关键开关")
    p.add_argument("--report_to", default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--local_rank", type=int, default=-1, help="injected by deepspeed launcher")
    args, _ = p.parse_known_args()
    return args


def format_alpaca(example):
    """Alpaca-style prompt format."""
    if example.get("input"):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": prompt}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if rank0:
        print(f"\n{'='*60}")
        print(f"LoRA SFT")
        print(f"  model:    {args.model_path}")
        print(f"  dataset:  {args.dataset_path}")
        print(f"  output:   {output_dir}")
        print(f"  seq_len:  {args.max_seq_length}")
        print(f"  bs/dev:   {args.per_device_batch_size}, grad_accum: {args.grad_accum_steps}")
        print(f"  lora:     r={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"  ds:       {args.deepspeed}")
        print(f"{'='*60}\n")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ★ HfDeepSpeedConfig: 必须在 from_pretrained 之前创建，让 transformers 走 zero.Init()
    # 否则 32B+ 模型会在每 rank 都 load 完整 weight → 单卡 24GB OOM
    dschf = None
    if args.deepspeed:
        if rank0:
            print(f"[{time.strftime('%H:%M:%S')}] init dist + setting up HfDeepSpeedConfig for ZeRO-3 sharded load")
        # 必须先 init dist group, 否则 DeepSpeedConfig.world_size=1 跟 train_batch_size 不一致 assert 失败
        import deepspeed as _ds
        _ds.init_distributed()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        import json as _json
        with open(args.deepspeed) as f:
            ds_config = _json.load(f)
        # 手动注入真实 batch / accum 值，避免 'auto' 在 HF Trainer ready 前不被替换
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_batch_size
        ds_config["gradient_accumulation_steps"] = args.grad_accum_steps
        ds_config["train_batch_size"] = args.per_device_batch_size * args.grad_accum_steps * world_size
        # ZeRO-3 bucket/threshold 必须是数字（不能 "auto"）
        if "zero_optimization" in ds_config:
            zo = ds_config["zero_optimization"]
            zo["reduce_bucket_size"] = int(5e8)
            zo["stage3_prefetch_bucket_size"] = int(5e8)
            zo["stage3_param_persistence_threshold"] = int(1e6)
        dschf = HfDeepSpeedConfig(ds_config)

    # model
    if rank0:
        print(f"[{time.strftime('%H:%M:%S')}] loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # DeepSpeed 处理 device 分配, 不要这里指定 device_map
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if rank0:
        print(f"[{time.strftime('%H:%M:%S')}] model loaded, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if rank0:
        model.print_trainable_parameters()

    # dataset
    if rank0:
        print(f"[{time.strftime('%H:%M:%S')}] loading dataset {args.dataset_path}")
    raw = load_dataset("json", data_files=args.dataset_path, split="train")
    ds = raw.map(format_alpaca, remove_columns=raw.column_names)
    if rank0:
        print(f"  examples: {len(ds)}, sample: {ds[0]['text'][:200]}...")

    # training args
    train_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=args.max_seq_length,
        report_to=args.report_to,
        # throughput observability
        log_level="info",
        logging_first_step=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tok,
    )

    start = time.time()
    trainer.train()
    dur = time.time() - start

    if rank0:
        print(f"\n[{time.strftime('%H:%M:%S')}] training done in {dur:.1f}s")
        print(f"  saving LoRA adapter only (skip 32B base gather) to {output_dir}")

    # 只 gather 可训练的 LoRA 参数（~268MB），不要 gather 整个 32B base (~64GB)。
    # ZeRO-3 默认会 AllGather 全部 weight → PCIe-only 8 卡 30min 超时
    if args.deepspeed:
        import deepspeed as _ds
        trainable = [p for p in model.parameters() if p.requires_grad]
        with _ds.zero.GatheredParameters(trainable, modifier_rank=0):
            if rank0:
                model.save_pretrained(str(output_dir))
                tok.save_pretrained(str(output_dir))
    else:
        if rank0:
            model.save_pretrained(str(output_dir))
            tok.save_pretrained(str(output_dir))

    if rank0:
        total_tokens = len(ds) * args.max_seq_length * args.num_epochs
        with open(output_dir / "throughput.json", "w") as f:
            json.dump({
                "model": args.model_path,
                "examples": len(ds),
                "epochs": args.num_epochs,
                "max_seq_length": args.max_seq_length,
                "total_tokens_est": total_tokens,
                "wall_time_sec": dur,
                "tokens_per_sec_est": total_tokens / dur,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            }, f, indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] save done")


if __name__ == "__main__":
    main()
