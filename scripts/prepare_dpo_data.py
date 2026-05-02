"""
合并多个 preference JSONL，去重并重新分割 train/val。

用法：
  python scripts/prepare_dpo_data.py \
    --input data/train/preference_all.jsonl data/synthesis_test/three_body_preference.jsonl \
    --output_dir data/train \
    --val_ratio 0.05
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output_dir", default="data/train")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for path in args.input:
        p = Path(path)
        if not p.exists():
            print(f"⚠️  找不到: {path}，跳过")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("instruction") and obj.get("input") and obj.get("chosen") and obj.get("rejected"):
                    records.append(obj)
        print(f"✅ 读取 {path}: 累计 {len(records)} 条")

    # 去重（instruction + input 相同视为重复）
    seen = set()
    deduped = []
    for r in records:
        key = r["instruction"] + r["input"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"✅ 去重后: {len(deduped)} 条（原始 {len(records)} 条）")

    random.seed(args.seed)
    random.shuffle(deduped)
    val_n = max(1, int(len(deduped) * args.val_ratio))
    train = deduped[val_n:]
    val   = deduped[:val_n]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "dpo_train.jsonl"
    val_path   = out / "dpo_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"📄 训练集: {len(train)} 条 → {train_path}")
    print(f"📄 验证集: {len(val)} 条 → {val_path}")


if __name__ == "__main__":
    main()
