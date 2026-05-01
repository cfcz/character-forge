"""角色状态提取器 — 从每章文本中抽取角色变化"""

from character_forge.extraction.chapter_splitter import Chapter
from character_forge.utils.llm import LLMClient

EXTRACTION_PROMPT = """\
你是一个专业的小说分析师。请仔细阅读以下小说章节，提取角色信息。

## 小说章节（第{chapter_index}章: {chapter_title}）

{chapter_content}

---

## 已知角色列表（之前章节出现过的）
{known_characters}

---

## 任务

请严格输出以下 JSON 格式，不要输出任何其他内容：

```json
{{
  "chapter_summary": "本章核心剧情的1-2句话摘要",
  "characters_appeared": ["本章出场的角色名"],
  "key_events": ["本章发生的关键事件，每个事件一句话"],
  "character_changes": [
    {{
      "name": "角色名",
      "new_facts_learned": ["该角色在本章新了解到的信息"],
      "unknown_facts_updated": ["本章新暴露出的、该角色目前尚不知道的重要信息（读者知道但角色不知道的）"],
      "emotional_shift": "情绪变化描述（如果没有变化写null）",
      "relationship_changes": [
        {{"target": "对方角色名", "change": "关系变化描述"}}
      ],
      "key_event": "对该角色最重要的事件（一句话）",
      "personality_development": "性格发展（如果有，否则null）"
    }}
  ],
  "new_characters": [
    {{
      "name": "新出场角色名",
      "identity": "一句话身份描述",
      "personality_traits": ["性格特征1", "性格特征2"],
      "speech_style": "说话风格描述"
    }}
  ]
}}
```

注意：
- 只提取重要角色的变化，路人不需要提取
- new_facts_learned：记录角色在本章亲眼/亲耳获知的信息
- unknown_facts_updated：记录本章揭示的、该角色尚不知情的重要信息（例如：别人对他的秘密计划、他被隐瞒的身世、幕后真相等）。这个字段必须出现在每个 character_changes 条目中；如果没有则填 []
- 如果某角色在本章没有变化，就不要把ta放在 character_changes 里
- new_characters 只包含首次出场的角色
"""

SCHEMA_RETRY_SUFFIX = """\

上一次输出缺少必需字段。请重新输出完整 JSON，并确保：
- 每个 character_changes 条目都包含 unknown_facts_updated 字段
- unknown_facts_updated 必须是数组；没有未知信息时写 []
- 不要省略任何 schema 字段
"""


class ChapterExtractor:
    """从单章文本中提取角色变化"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(
        self,
        chapter: Chapter,
        known_characters: list[str],
    ) -> dict:
        """
        提取单章的角色变化。

        Args:
            chapter: 章节对象
            known_characters: 之前章节已经出现过的角色名列表

        Returns:
            结构化的章节分析结果（dict）
        """
        known_str = (
            "、".join(known_characters) if known_characters else "（这是第一章，暂无已知角色）"
        )

        # 如果章节太长，截断（保留首尾，中间摘要）
        content = self._truncate_if_needed(chapter.content, max_chars=8000)

        prompt = EXTRACTION_PROMPT.format(
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            chapter_content=content,
            known_characters=known_str,
        )

        result = self.llm.generate_json(prompt)
        result, stats = self._normalize_extraction(result)

        if self._should_retry_for_missing_unknown(stats):
            retry_result = self.llm.generate_json(prompt + SCHEMA_RETRY_SUFFIX)
            retry_result, retry_stats = self._normalize_extraction(retry_result)
            if retry_stats["missing_unknown_fields"] < stats["missing_unknown_fields"]:
                result, stats = retry_result, retry_stats

        result["_validation"] = stats
        return result

    def _normalize_extraction(self, result: dict) -> tuple[dict, dict]:
        """补齐 LLM 可能省略的字段，并返回 schema 质量统计。"""
        if not isinstance(result, dict):
            result = {}

        result.setdefault("chapter_summary", "")
        result.setdefault("characters_appeared", [])
        result.setdefault("key_events", [])
        result.setdefault("character_changes", [])
        result.setdefault("new_characters", [])

        changes = result.get("character_changes")
        if not isinstance(changes, list):
            changes = []
            result["character_changes"] = changes

        missing_unknown_fields = 0
        unknown_fact_count = 0
        for change in changes:
            if not isinstance(change, dict):
                continue

            if "unknown_facts_updated" not in change:
                missing_unknown_fields += 1
                change["unknown_facts_updated"] = []
            elif not isinstance(change["unknown_facts_updated"], list):
                change["unknown_facts_updated"] = []

            change.setdefault("new_facts_learned", [])
            change.setdefault("relationship_changes", [])
            change.setdefault("emotional_shift", None)
            change.setdefault("key_event", "")
            change.setdefault("personality_development", None)

            unknown_fact_count += len(change["unknown_facts_updated"])

        stats = {
            "character_changes": len(changes),
            "missing_unknown_fields": missing_unknown_fields,
            "unknown_facts_updated": unknown_fact_count,
        }
        return result, stats

    def _should_retry_for_missing_unknown(self, stats: dict) -> bool:
        return (
            stats["character_changes"] > 0
            and stats["missing_unknown_fields"] == stats["character_changes"]
        )

    def _truncate_if_needed(self, text: str, max_chars: int = 8000) -> str:
        if len(text) <= max_chars:
            return text
        # 保留前 40% + 后 40%，中间标记省略
        head_len = int(max_chars * 0.4)
        tail_len = int(max_chars * 0.4)
        return (
            text[:head_len]
            + f"\n\n[...中间省略约{len(text) - head_len - tail_len}字...]\n\n"
            + text[-tail_len:]
        )
