"""主 Pipeline — 输入小说 → 章节切分 → 逐章提取 → 构建角色状态"""

import json
import time
from pathlib import Path

from character_forge.extraction.chapter_splitter import split_chapters
from character_forge.extraction.chapter_extractor import ChapterExtractor
from character_forge.memory.state_manager import CharacterStateManager
from character_forge.utils.llm import LLMClient


class NovelProcessor:
    """小说处理主流程"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.extractor = ChapterExtractor(llm)
        self.state_manager = CharacterStateManager(llm)

    def process(
        self,
        novel_text: str,
        max_chapters: int | None = None,
        verbose: bool = True,
        checkpoint_path: str | None = None,   # 每章处理完后自动保存到这个路径
    ) -> CharacterStateManager:
        """
        处理一本小说，返回包含所有角色状态的 manager。

        Args:
            novel_text: 小说全文
            max_chapters: 最多处理多少章（调试用）
            verbose: 是否打印进度
            checkpoint_path: 每章处理完后自动保存到这个路径，下次运行时自动从断点续跑
        """
        # 1. 切分章节
        chapters = split_chapters(novel_text)
        if max_chapters:
            chapters = chapters[:max_chapters]

        if verbose:
            print(f"📖 共切分出 {len(chapters)} 个章节")
            for ch in chapters:
                print(f"   第{ch.index}章: {ch.title} ({ch.char_count}字)")
            print()

        # 2. 如果有 checkpoint，尝试从上次中断处续跑
        start_from_index = 0
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.state_manager = CharacterStateManager.load(checkpoint_path, self.llm)
                # 找到已处理的最大章节号
                processed_chapters = {r["chapter"] for r in self.state_manager._history}
                if processed_chapters:
                    last_done = max(processed_chapters)
                    # 找到下一个未处理章节在列表中的位置
                    start_from_index = next(
                        (i for i, ch in enumerate(chapters) if ch.index > last_done),
                        len(chapters),  # 全部处理完了
                    )
                    if verbose:
                        print(f"♻️  从 checkpoint 恢复：已完成到第{last_done}章，"
                              f"从第{chapters[start_from_index].index}章继续\n"
                              if start_from_index < len(chapters)
                              else f"♻️  从 checkpoint 恢复：所有章节已处理完毕\n")
            except Exception as e:
                print(f"⚠️ 加载 checkpoint 失败 ({e})，从头开始处理")
                start_from_index = 0

        # 3. 逐章处理
        for ch in chapters[start_from_index:]:
            if verbose:
                print(f"🔍 正在分析第{ch.index}章: {ch.title} ...")

            start_time = time.time()

            # 提取
            try:
                extraction = self.extractor.extract(
                    chapter=ch,
                    known_characters=self.state_manager.list_characters(),
                )
            except Exception as e:
                print(f"   ⚠️ 提取失败: {e}，跳过本章")
                continue

            # 更新状态
            self.state_manager.update_from_chapter(ch.index, extraction)

            elapsed = time.time() - start_time
            if verbose:
                n_chars = len(extraction.get("characters_appeared", []))
                n_changes = len(extraction.get("character_changes", []))
                n_new = len(extraction.get("new_characters", []))
                print(f"   ✅ {elapsed:.1f}s | "
                      f"出场{n_chars}人 | 变化{n_changes}条 | 新角色{n_new}个")

            # 每章处理完立即保存 checkpoint
            if checkpoint_path:
                try:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    self.state_manager.save(checkpoint_path)
                    if verbose:
                        print(f"   💾 checkpoint 已保存 ({checkpoint_path})")
                except Exception as e:
                    print(f"   ⚠️ checkpoint 保存失败: {e}")

        # 3. 打印结果摘要
        if verbose:
            print(f"\n{'='*50}")
            print(f"📊 处理完成！共识别 {len(self.state_manager.list_characters())} 个角色：")
            for name in self.state_manager.list_characters():
                state = self.state_manager.get_state(name)
                print(f"\n{'─'*40}")
                print(f"👤 {state.name} (截止第{state.chapter}章)")
                print(f"   身份: {state.identity}")
                print(f"   性格: {', '.join(state.personality_traits[:5])}")
                print(f"   情绪: {state.emotional_state}")
                print(f"   已知信息: {len(state.known_facts)}条")
                print(f"   未知信息: {len(state.unknown_facts)}条")
                print(f"   关键记忆: {len(state.key_memories)}条")
                print(f"   关系: {len(state.relationships)}段")

        return self.state_manager


def process_novel_file(
    file_path: str,
    output_path: str = "character_states.json",
    provider: str = "deepseek",
    max_chapters: int | None = None,
):
    """便捷函数：处理小说文件并保存结果"""
    text = Path(file_path).read_text(encoding="utf-8")
    llm = LLMClient(provider=provider)
    processor = NovelProcessor(llm)
    manager = processor.process(text, max_chapters=max_chapters)
    manager.save(output_path)
    print(f"\n💾 角色状态已保存到 {output_path}")
    return manager


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m character_forge.pipeline <小说文件路径> [输出路径] [max_chapters]")
        print("示例: python -m character_forge.pipeline data/examples/demo.txt output.json 5")
        sys.exit(1)

    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "character_states.json"
    max_chapters = int(sys.argv[3]) if len(sys.argv) > 3 else None

    process_novel_file(file_path, output_path, max_chapters=max_chapters)
