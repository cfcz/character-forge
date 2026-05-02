"""
Character Forge — GRPO 训练脚本
================================
基于 trl 的 GRPOTrainer，不依赖 LLaMA-Factory。

流程：
  1. 从 adapter_config.json 读取 base model 路径
  2. 加载 base model + SFT LoRA → merge 成一个干净的起点
  3. 在 merged model 上加新 LoRA，进行 GRPO 训练
  4. 保存最终 adapter

用法：
  cd /root/character-forge
  python scripts/train_grpo.py \
    --sft_model /root/output/qwen_sft_full \
    --train_data data/train/grpo_train.jsonl \
    --output_dir /root/output/qwen_grpo
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from character_forge.training.reward_function import character_reward_fn


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_merged_model(sft_model_path: str):
    """
    从 SFT adapter 目录加载模型：
      1. 读 adapter_config.json 拿到 base model 路径
      2. 加载 base model
      3. 套上 SFT LoRA → merge_and_unload()
    返回 (merged_model, tokenizer)
    """
    adapter_config_path = Path(sft_model_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config["base_model_name_or_path"]
    print(f"📦 Base model: {base_model_path}")
    print(f"📦 SFT adapter: {sft_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path,       # tokenizer 存在 SFT 输出目录里
        trust_remote_code=True,
    )

    print("🔄 加载 base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("🔄 合并 SFT LoRA...")
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    model = model.merge_and_unload()
    model.eval()
    print("✅ SFT LoRA 已合并")

    return model, tokenizer


def build_dataset(tokenizer, data_path: str) -> Dataset:
    """
    加载 GRPO 数据，格式化成 chat prompt。
    保留 instruction 列供 reward function 使用。
    """
    raw = load_jsonl(data_path)
    formatted = []
    for item in raw:
        instruction = item.get("instruction", "")
        input_text  = item.get("input", "")
        if not instruction or not input_text:
            continue

        # 格式化为 Qwen chat 格式
        messages = [
            {"role": "system",  "content": instruction},
            {"role": "user",    "content": input_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        formatted.append({
            "prompt":      prompt,
            "instruction": instruction,   # reward function 需要这个字段
            "input":       input_text,
        })

    print(f"✅ 数据集加载完成：{len(formatted)} 条")
    return Dataset.from_list(formatted)


# ── 主训练流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Character Forge GRPO Training")
    parser.add_argument("--sft_model",   default="/root/output/qwen_sft_full",
                        help="SFT adapter 目录")
    parser.add_argument("--train_data",  default="/root/character-forge/data/train/grpo_train.jsonl",
                        help="GRPO 训练数据路径")
    parser.add_argument("--output_dir",  default="/root/output/qwen_grpo",
                        help="输出目录")
    parser.add_argument("--num_epochs",       type=int,   default=2)
    parser.add_argument("--num_generations",  type=int,   default=8,
                        help="每个 prompt 采样的回答数 G（默认 8）")
    parser.add_argument("--learning_rate",    type=float, default=5e-6)
    parser.add_argument("--max_new_tokens",   type=int,   default=256)
    args = parser.parse_args()

    # ── 1. 加载模型 ───────────────────────────────────────────────────────────
    model, tokenizer = load_merged_model(args.sft_model)

    # trl/transformers 版本兼容：部分 transformers 版本不会在 __init__ 里设置
    # warnings_issued，但 trl GRPOTrainer 在初始化时会访问它
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ── 2. 加载数据 ───────────────────────────────────────────────────────────
    train_dataset = build_dataset(tokenizer, args.train_data)

    # ── 3. 新 LoRA 配置（GRPO 阶段） ──────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 4. GRPO 训练配置 ──────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,

        # 核心 GRPO 参数
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,   # 注意：此版本 trl 用 max_completion_length

        # 训练超参
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,  # 等效 batch = num_generations
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,

        # 关键：保留 instruction 列，reward function 需要它解析角色状态
        remove_unused_columns=False,

        # 日志与保存
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    # ── 5. 训练 ───────────────────────────────────────────────────────────────
    print("\n🚀 开始 GRPO 训练...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[character_reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    # ── 6. 保存 ───────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ 训练完成！模型已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
