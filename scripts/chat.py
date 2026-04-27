"""命令行对话测试脚本

用法:
  # 先跑 pipeline 生成角色状态
  python scripts/run_demo.py

  # 再用这个脚本和角色对话
  python scripts/chat.py --states data/output_states.json

  # 指定角色和章节
  python scripts/chat.py --states data/output_states.json --character 沈夜 --chapter 3
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from character_forge.memory.state_manager import CharacterStateManager
from character_forge.agent.character_agent import CharacterAgent
from character_forge.utils.llm import LLMClient, LocalLLMClient


def main():
    parser = argparse.ArgumentParser(description="CharacterForge 对话测试")
    parser.add_argument("--states", default="data/output_states.json", help="角色状态文件")
    parser.add_argument("--character", default=None, help="要对话的角色名")
    parser.add_argument("--chapter", type=int, default=None, help="对话的时间点（第几章）")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--local-model", default=None, help="本地模型路径（SFT 训练后），指定此项则不调用 API")
    parser.add_argument("--show-reasoning", action="store_true", default=True, help="显示推理过程")
    args = parser.parse_args()

    # 加载角色状态
    states_path = Path(args.states)
    if not states_path.exists():
        print(f"❌ 找不到角色状态文件: {states_path}")
        print("请先运行: python scripts/run_demo.py")
        return

    if args.local_model:
        llm = LocalLLMClient(model_path=args.local_model)
    else:
        llm = LLMClient(provider=args.provider)
    manager = CharacterStateManager.load(states_path, llm)

    characters = manager.list_characters()
    if not characters:
        print("❌ 没有找到任何角色")
        return

    # 选择角色
    character_name = args.character
    if not character_name:
        print(f"\n📋 可对话的角色：")
        for i, name in enumerate(characters, 1):
            state = manager.get_state(name)
            print(f"  {i}. {name} — {state.identity[:30]}...")
        print()
        choice = input("请输入角色编号或名字: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            character_name = characters[idx] if 0 <= idx < len(characters) else characters[0]
        else:
            character_name = choice

    # 获取角色状态
    if args.chapter:
        state = manager.get_state_at_chapter(character_name, args.chapter)
        time_label = f"第{args.chapter}章"
    else:
        state = manager.get_state(character_name)
        time_label = f"第{state.chapter}章（最终状态）"

    if not state:
        print(f"❌ 找不到角色: {character_name}")
        return

    # 创建 Agent
    agent = CharacterAgent(state, llm, enable_leak_check=True)

    # 开始对话
    print(f"\n{'='*55}")
    print(f"🎭 现在与【{state.name}】对话")
    print(f"   时间点：{time_label}")
    print(f"   身份：{state.identity}")
    print(f"   情绪：{state.emotional_state}")
    print(f"{'='*55}")
    print("输入 'q' 退出 | 输入 'r' 切换是否显示推理过程 | 输入 'reset' 清空历史\n")

    show_reasoning = args.show_reasoning

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("✅ 对话历史已清空\n")
            continue
        if user_input.lower() == "r":
            show_reasoning = not show_reasoning
            print(f"✅ 推理过程显示: {'开启' if show_reasoning else '关闭'}\n")
            continue

        result = agent.chat(user_input)

        # 显示推理过程
        if show_reasoning and result.reasoning:
            print(f"\n💭 推理过程：")
            for line in result.reasoning.split("\n"):
                if line.strip():
                    print(f"   {line}")
            print()

        # 信息泄露警告
        if result.leaked:
            print(f"⚠️  [检测到信息边界违规，已重新生成]")
            print(f"   原因: {result.leak_reason}\n")

        print(f"{state.name}：{result.response}\n")


if __name__ == "__main__":
    main()
