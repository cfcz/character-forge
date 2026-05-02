"""
把旧的 preference JSONL 转成 GRPO 训练格式。

GRPO 不需要 chosen/rejected，只需要 prompt（instruction + input）。
输出格式与 LLaMA-Factory 的 alpaca 格式一致：
  {"instruction": "...", "input": "...", "output": ""}

用法：
  python scripts/prepare_grpo_data.py \\
    --input data/train/preference_all.jsonl \\
    --output data/train/grpo_train.jsonl

也可以合并多个文件：
  python scripts/prepare_grpo_data.py \\
    --input data/synthesis/heros_preference.jsonl data/synthesis/three_body_preference.jsonl \\
    --output data/train/grpo_train.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def convert(input_paths: list[Path], output_path: Path, val_ratio: float = 0.05, seed: int = 42):
    records = []
    for path in input_paths:
        if not path.exists():
            print(f"⚠️  找不到文件: {path}，跳过")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # preference 格式：instruction / input / chosen / rejected
                # SFT 格式：instruction / input / output
                # 两种都兼容
                instruction = obj.get("instruction", "")
                input_text  = obj.get("input", "")

                if not instruction or not input_text:
                    continue

                records.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": "",   # GRPO 不需要 output，模型自己生成
                })

    if not records:
        print("❌ 没有读取到任何有效数据")
        return

    # 去重（instruction + input 相同的视为重复）
    seen = set()
    deduped = []
    for r in records:
        key = r["instruction"] + r["input"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"✅ 去重后剩余 {len(deduped)} 条（原始 {len(records)} 条）")

    # 打乱并分割 train/val
    random.seed(seed)
    random.shuffle(deduped)
    val_n = max(1, int(len(deduped) * val_ratio))
    train_records = deduped[val_n:]
    val_records   = deduped[:val_n]

    # 写入
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = output_path.with_name(output_path.stem + "_val" + output_path.suffix)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"📄 训练集：{len(train_records)} 条 → {output_path}")
    print(f"📄 验证集：{len(val_records)} 条 → {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Preference → GRPO 数据转换")
    parser.add_argument("--input",  nargs="+", required=True, help="输入 JSONL 文件（可多个）")
    parser.add_argument("--output", required=True, help="输出训练集路径")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例（默认 0.05）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)
    convert(input_paths, output_path, val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
