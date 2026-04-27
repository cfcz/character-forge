"""
数据合成入口脚本

用法：

  # 从已有 JSON 加载，生成 SFT 数据
  python scripts/synthesize_data.py \
    --from-json data/output_states.json \
    --novel data/examples/three_body.txt \
    --mode sft

  # 生成 Preference/GRPO 数据
  python scripts/synthesize_data.py \
    --from-json data/output_states.json \
    --novel data/examples/three_body.txt \
    --mode preference

  # 两种都生成
  python scripts/synthesize_data.py \
    --from-json data/output_states.json \
    --novel data/examples/three_body.txt \
    --mode both
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from character_forge.memory.state_manager import CharacterStateManager
from character_forge.pipeline import NovelProcessor
from character_forge.synthesis.data_synthesizer import (
    DataSynthesizer,
    save_jsonl,
    to_preference_format,
    to_sft_format,
)
from character_forge.utils.llm import LLMClient


def _preview_sft(items):
    for i in range(min(5, len(items))):
        first = items[i]
        print("\n" + "=" * 60)
        print(f"📄 SFT 样本预览（第{i + 1}条）：")
        print("=" * 60)
        print(f"角色：{first.character}（第{first.chapter}章）")
        print(f"用户问题：{first.question}")
        print(f"\n✅ 回答：\n{first.answer}")
        print("=" * 60)


def _preview_preference(items):
    first = items[0]
    print("\n" + "=" * 60)
    print("📄 Preference 样本预览（第1条）：")
    print("=" * 60)
    print(f"角色：{first.character}（第{first.chapter}章）")
    print(f"触发的 unknown_fact：{first.unknown_fact}")
    print(f"用户问题：{first.question}")
    print(f"\n✅ chosen（守住边界）：\n{first.chosen}")
    print(f"\n❌ rejected（泄露信息）：\n{first.rejected}")
    print("=" * 60)


def _select_top_characters(manager, top_n: int, novel_text: str) -> list[str]:
    """
    按角色丰富度评分，自动选取前 top_n 个角色。

    评分公式（越高越适合合成）：
      known_facts × 3 + key_memories × 2 + unknown_facts × 4
      + relationships × 1 + dialogue_count × 2

    unknown_facts 权重最高：边界越清晰，Preference 数据越好。
    dialogue_count 通过在原文中搜索角色名出现次数近似估算。
    """
    scores: list[tuple[str, int]] = []
    for name in manager.list_characters():
        state = manager.get_state(name)
        if state is None:
            continue
        # 在原文中粗略统计角色名出现频率（近似台词/存在感）
        dialogue_count = novel_text.count(name)
        score = (
            len(state.known_facts) * 3
            + len(state.key_memories) * 2
            + len(state.unknown_facts) * 4
            + len(state.relationships) * 1
            + min(dialogue_count, 200) * 2   # 上限 200，避免主角过度主导
        )
        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 角色丰富度排名（取前 {top_n} 个）：")
    for rank, (name, score) in enumerate(scores[:top_n], 1):
        state = manager.get_state(name)
        print(f"   {rank:2d}. {name:<10} 分={score:4d} "
              f"| 已知{len(state.known_facts)}条 未知{len(state.unknown_facts)}条 "
              f"记忆{len(state.key_memories)}条 原文出现{novel_text.count(name)}次")

    return [name for name, _ in scores[:top_n]]


def main():
    parser = argparse.ArgumentParser(description="CharacterForge 数据合成")

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-json", type=str, help="从已有 JSON 加载角色状态（推荐，省 API 费用）")

    parser.add_argument("--novel", type=str, required=True, help="小说原文路径（用于台词提取 few-shot）")
    parser.add_argument("--max-chapters", type=int, default=None, help="提取时最多处理几章（仅在不用 --from-json 时有效）")

    parser.add_argument("--mode", choices=["sft", "preference", "both"], default="both", help="生成哪种数据（默认 both）")
    parser.add_argument("--characters", nargs="+", default=None, help="只处理这些角色（默认全部）")
    parser.add_argument("--top-n", type=int, default=None, help="按角色丰富度自动选取前 N 个角色（与 --characters 互斥）")
    parser.add_argument("--questions-per-fact", type=int, default=2, help="每条 unknown_fact 生成几个问题（默认 2）")
    parser.add_argument("--sft-questions-per-slice", type=int, default=4, help="每个章节切片生成几个 SFT 问题（默认 4）")
    parser.add_argument("--chapter-stride", type=int, default=5, help="每隔几章取一个状态切片（默认 5）")
    parser.add_argument("--dialogue-examples", type=int, default=8, help="每个角色提取多少条原著台词示例（默认 8）")
    parser.add_argument("--dry-run", type=int, default=None, help="只保留前 N 条样本，用于快速验证质量")

    parser.add_argument("--output-dir", type=str, default="data/synthesis", help="输出目录（默认 data/synthesis）")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "qwen", "openai"])
    parser.add_argument("--model", type=str, default=None, help="可选：显式指定模型名，例如 deepseek-v4-flash 或 deepseek-v4-pro")

    args = parser.parse_args()

    print("🔧 初始化 LLM...")
    llm = LLMClient(provider=args.provider, model=args.model)

    novel_path = Path(args.novel)
    if not novel_path.exists():
        print(f"❌ 找不到小说文件: {novel_path}")
        sys.exit(1)
    novel_text = novel_path.read_text(encoding="utf-8")
    print(f"📖 已读取小说: {novel_path.name} ({len(novel_text)} 字)")

    if args.from_json:
        json_path = Path(args.from_json)
        if not json_path.exists():
            print(f"❌ 找不到 JSON 文件: {json_path}")
            sys.exit(1)
        print(f"📂 从 JSON 加载角色状态: {json_path}")
        manager = CharacterStateManager.load(json_path, llm)
        print(f"   角色数: {len(manager.list_characters())}，历史章节数: {len(manager._history)}")
    else:
        print("🔍 直接从小说提取角色状态（会调用 LLM API）...")
        processor = NovelProcessor(llm)
        manager = processor.process(novel_text, max_chapters=args.max_chapters)

    # 角色筛选：--characters 手动指定 > --top-n 自动排名 > 全部
    selected_characters = args.characters
    if selected_characters is None and args.top_n:
        selected_characters = _select_top_characters(manager, args.top_n, novel_text)

    print(f"\n📋 将处理角色: {selected_characters or manager.list_characters()}")

    synthesizer = DataSynthesizer(
        llm=llm,
        n_questions_per_fact=args.questions_per_fact,
        chapter_stride=args.chapter_stride,
        sft_questions_per_slice=args.sft_questions_per_slice,
        dialogue_examples=args.dialogue_examples,
    )

    output_dir = Path(args.output_dir)
    novel_name = novel_path.stem
    generated_any = False

    if args.mode in {"sft", "both"}:
        if args.dry_run:
            print(f"\n🚀 开始合成 SFT 数据（dry-run，最多 {args.dry_run} 条）...\n")
        else:
            print("\n🚀 开始合成 SFT 数据...\n")
        sft_items = synthesizer.synthesize_sft(
            manager=manager,
            novel_text=novel_text,
            characters=selected_characters,
            max_chapters=args.max_chapters,
            max_samples=args.dry_run,
        )
        if sft_items:
            generated_any = True
            print(f"\n✅ 共生成 {len(sft_items)} 条 SFT 原始样本")
            _preview_sft(sft_items)
            sft_records = to_sft_format(sft_items, manager)
            sft_path = output_dir / f"{novel_name}_sft.jsonl"
            save_jsonl(sft_records, sft_path)
        else:
            print("⚠️ 没有生成任何 SFT 样本")

    if args.mode in {"preference", "both"}:
        if args.dry_run:
            print(f"\n🚀 开始合成 Preference 数据（dry-run，最多 {args.dry_run} 条）...\n")
        else:
            print("\n🚀 开始合成 Preference 数据...\n")
        pref_items = synthesizer.synthesize_preference(
            manager=manager,
            novel_text=novel_text,
            characters=selected_characters,
            max_chapters=args.max_chapters,
            max_samples=args.dry_run,
        )
        if pref_items:
            generated_any = True
            print(f"\n✅ 共生成 {len(pref_items)} 条 Preference 原始样本")
            _preview_preference(pref_items)
            pref_records = to_preference_format(pref_items, manager)
            pref_path = output_dir / f"{novel_name}_preference.jsonl"
            save_jsonl(pref_records, pref_path)
        else:
            print("⚠️ 没有生成任何 Preference 样本，请检查角色状态是否包含 unknown_facts")

    if not generated_any:
        sys.exit(1)

    print("\n🎉 完成！")


if __name__ == "__main__":
    main()
