"""快速测试脚本 — 用 demo 小说跑通全链路

用法:
  export DEEPSEEK_API_KEY="你的key"
  python scripts/run_demo.py

也支持其他 provider:
  export DASHSCOPE_API_KEY="你的key"
  python scripts/run_demo.py --provider qwen
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path

# 确保从项目根目录 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from character_forge.pipeline import NovelProcessor
from character_forge.utils.llm import LLMClient


def main():
    parser = argparse.ArgumentParser(description="CharacterForge Demo")
    parser.add_argument("--novel", default="data/examples/three_body.txt", help="小说文件路径")
    parser.add_argument("--output", default="data/output_states.json", help="输出文件路径")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "openai"])
    parser.add_argument("--model", type=str, default=None, help="可选：显式指定模型名")
    parser.add_argument("--max-chapters", type=int, default=None, help="最多处理多少章（调试用）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 路径，每章处理完自动保存；若文件已存在则从上次中断处续跑。"
                             "不指定则不保存 checkpoint。建议设为 --output 同名文件（例如 data/output_states.json）")
    args = parser.parse_args()

    print("🎭 CharacterForge — 小说角色状态提取系统")
    print("=" * 50)

    # 1. 读取小说
    novel_path = Path(args.novel)
    if not novel_path.exists():
        print(f"❌ 文件不存在: {novel_path}")
        return
    text = novel_path.read_text(encoding="utf-8")
    print(f"📖 已读取小说: {novel_path.name} ({len(text)}字)")

    # 2. 初始化
    llm = LLMClient(provider=args.provider, model=args.model)
    processor = NovelProcessor(llm)

    # 3. 处理（支持 checkpoint 断点续跑）
    checkpoint = args.checkpoint or args.output   # 默认用 output 路径作为 checkpoint
    manager = processor.process(
        text,
        max_chapters=args.max_chapters,
        checkpoint_path=checkpoint,
    )

    # 4. 保存最终结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manager.save(output_path)
    print(f"\n💾 已保存到 {output_path}")

    # 5. 展示角色状态快照
    print(f"\n{'='*50}")
    print("📋 角色状态快照（可直接注入 prompt）：")
    for name in manager.list_characters():
        state = manager.get_state(name)
        print(f"\n{'━'*50}")
        print(state.to_prompt())


if __name__ == "__main__":
    main()
