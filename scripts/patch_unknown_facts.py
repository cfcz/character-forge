"""
补丁脚本：对已有 JSON 中缺少 unknown_facts 的角色，用 LLM 补充生成。

用法：
  python scripts/patch_unknown_facts.py \
    --json data/three_body_full.json \
    --novel data/examples/three_body.txt \
    --provider deepseek

原理：
  对每个 unknown_facts 为空的角色，让 LLM 根据角色的 known_facts、
  key_memories、relationships 以及小说简介，推断出"读者知道但角色尚不知情"的重要信息。
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from character_forge.memory.state_manager import CharacterStateManager
from character_forge.utils.llm import LLMClient


UNKNOWN_FACTS_PROMPT = """\
你是一位熟悉小说《{novel_name}》的分析师。

## 角色：{name}
身份：{identity}
截止第{chapter}章的状态：
- 已知信息：{known_facts}
- 关键记忆：{key_memories}
- 当前情绪：{emotional_state}
- 当前目标：{goals}
- 关系网络：{relationships}

## 任务
请根据这个角色在第{chapter}章时的认知状态，列出**读者/旁观者知道、但该角色本人尚不知情**的重要信息。

这些"未知信息"是角色的知识盲区，典型包括：
- 其他角色对他的隐瞒或欺骗
- 他尚未意识到的真相、阴谋或危险
- 他被隐瞒的身世、关系或秘密
- 将要发生但他预料不到的重大事件

严格输出 JSON，不要输出其他内容：
```json
{{
  "unknown_facts": [
    "未知信息1（一句话，以'该角色不知道……'或'……对该角色隐瞒了……'等形式描述）",
    "未知信息2",
    "未知信息3"
  ]
}}
```

要求：
- 只写该角色在第{chapter}章时真实存在的知识盲区，不要凭空捏造
- 最多8条，优先挑对后续剧情影响最大的
- 如果确实没有重要盲区，返回空列表
"""


def patch_character(name: str, state, llm: LLMClient, novel_name: str) -> list[str]:
    rels = [f"{r.target}({r.relation_type})" for r in state.relationships[:5]]
    prompt = UNKNOWN_FACTS_PROMPT.format(
        novel_name=novel_name,
        name=name,
        chapter=state.chapter,
        identity=state.identity or "未知",
        known_facts=json.dumps(state.known_facts[:10], ensure_ascii=False),
        key_memories=json.dumps(state.key_memories[:5], ensure_ascii=False),
        emotional_state=state.emotional_state or "未知",
        goals=json.dumps(state.goals[:3], ensure_ascii=False),
        relationships=json.dumps(rels, ensure_ascii=False),
    )
    try:
        result = llm.generate_json(prompt)
        return result.get("unknown_facts") or []
    except Exception as e:
        print(f"   ⚠️ 生成失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="补充 unknown_facts")
    parser.add_argument("--json", required=True, help="角色状态 JSON 路径")
    parser.add_argument("--novel", required=True, help="小说文件路径（用于获取书名）")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-known-facts", type=int, default=2,
                        help="至少有这么多 known_facts 才值得生成 unknown_facts（默认 2）")
    parser.add_argument("--overwrite", action="store_true",
                        help="对已有 unknown_facts 的角色也重新生成")
    args = parser.parse_args()

    llm = LLMClient(provider=args.provider, model=args.model)
    novel_name = Path(args.novel).stem

    json_path = Path(args.json)
    manager = CharacterStateManager.load(json_path, llm)

    chars = manager.list_characters()
    print(f"📂 已加载 {len(chars)} 个角色")

    patched = 0
    for name in chars:
        state = manager.characters[name]
        already_has = len(state.unknown_facts) > 0

        if already_has and not args.overwrite:
            print(f"⏭  {name}：已有 {len(state.unknown_facts)} 条 unknown_facts，跳过")
            continue
        if len(state.known_facts) < args.min_known_facts:
            print(f"⏭  {name}：known_facts 不足 {args.min_known_facts} 条，跳过")
            continue

        print(f"🔍 {name}（截止第{state.chapter}章，{len(state.known_facts)}条已知信息）...")
        unknown_facts = patch_character(name, state, llm, novel_name)
        state.unknown_facts = unknown_facts
        print(f"   ✅ 生成 {len(unknown_facts)} 条 unknown_facts")
        patched += 1

    manager.save(json_path)
    print(f"\n💾 已保存，共补充 {patched} 个角色的 unknown_facts → {json_path}")


if __name__ == "__main__":
    main()
