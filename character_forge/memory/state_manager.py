"""角色状态管理器 — 增量构建 + 压缩

这是系统的算法核心之一：
- 每处理一章，就把该章的角色变化合并到全局状态中
- 当状态膨胀到一定程度，触发压缩（episodic → semantic）
- 压缩时保留关键信息，丢弃冗余细节
"""

import json
import copy
from pathlib import Path

from character_forge.schema.character_state import (
    CharacterState,
    RelationshipState,
)
from character_forge.utils.llm import LLMClient


COMPRESSION_PROMPT = """\
你是一个记忆管理专家。以下是小说角色 "{name}" 截止第{chapter}章的累积状态信息。
信息量已经过大，请帮我压缩。

## 当前状态
{current_state}

## 压缩要求

请将以上信息压缩为更精简的版本，严格输出JSON：

```json
{{
  "identity": "一句话身份描述（保持不变或更新）",
  "personality_traits": ["压缩后的核心性格特征，最多5个"],
  "speech_style": "说话风格（一句话）",
  "known_facts": ["保留最重要的已知信息，最多15条，合并可合并的"],
  "unknown_facts": ["保留最重要的未知信息，最多10条"],
  "emotional_state": "当前情绪",
  "goals": ["当前最重要的目标，最多3个"],
  "key_memories": ["对人格影响最大的记忆，最多8条，合并可合并的"],
  "relationships": [
    {{"target": "角色名", "relation_type": "关系类型", "description": "描述", "trust_level": "信任度"}}
  ]
}}
```

压缩原则：
1. **已知信息**：合并相关条目（如"知道A"和"知道A的原因"→合并为一条），丢弃琐碎信息
2. **关键记忆**：只保留真正改变了角色性格或行为的事件
3. **关系**：只保留有实质影响的关系
4. **绝对不能丢弃**：未知信息（这是信息边界的关键），当前目标
"""


class CharacterStateManager:
    """管理所有角色的状态，支持增量更新和压缩"""

    def __init__(
        self,
        llm: LLMClient,
        compress_threshold: int = 20,
        enable_compression: bool = True,
    ):
        """
        Args:
            llm: LLM客户端
            compress_threshold: known_facts 超过这个数量时触发压缩
        """
        self.llm = llm
        self.compress_threshold = compress_threshold
        self.enable_compression = enable_compression
        self.characters: dict[str, CharacterState] = {}
        self._history: list[dict] = []  # 保存每章的原始提取结果

    def update_from_chapter(self, chapter_index: int, extraction: dict):
        """根据单章提取结果更新所有角色状态"""

        self._history.append({"chapter": chapter_index, "extraction": extraction})

        # 处理新角色
        for new_char in (extraction.get("new_characters") or []):
            name = new_char.get("name", "")
            if not name:
                continue
            if name not in self.characters:
                self.characters[name] = CharacterState(
                    name=name,
                    chapter=chapter_index,
                    identity=new_char.get("identity", ""),
                    personality_traits=new_char.get("personality_traits") or [],
                    speech_style=new_char.get("speech_style", ""),
                    known_facts=[],
                    unknown_facts=[],
                    emotional_state="初始状态",
                    goals=[],
                )

        # 处理角色变化
        for change in (extraction.get("character_changes") or []):
            name = change["character_name"] if "character_name" in change else change.get("name", "")
            if not name:
                continue

            # 如果角色还没建档，先创建一个基础档案
            if name not in self.characters:
                self.characters[name] = CharacterState(
                    name=name,
                    chapter=chapter_index,
                    identity="",
                    personality_traits=[],
                    speech_style="",
                    known_facts=[],
                    unknown_facts=[],
                    emotional_state="",
                    goals=[],
                )

            state = self.characters[name]
            state.chapter = chapter_index

            # 合并新获知的信息
            for fact in (change.get("new_facts_learned") or []):
                if fact and fact not in state.known_facts:
                    state.known_facts.append(fact)

            # 合并新暴露的未知信息（角色尚不知情的重要事实）
            for uf in (change.get("unknown_facts_updated") or []):
                if uf and uf not in state.unknown_facts:
                    state.unknown_facts.append(uf)

            # 更新情绪
            if change.get("emotional_shift"):
                state.emotional_state = change["emotional_shift"]

            # 更新关系
            for rel_change in (change.get("relationship_changes") or []):
                self._update_relationship(state, rel_change)

            # 添加关键记忆
            if change.get("key_event"):
                state.key_memories.append(
                    f"[第{chapter_index}章] {change['key_event']}"
                )

            # 性格发展：过滤掉 null 字符串、空值、重复项、超长描述句
            dev = change.get("personality_development")
            if (
                dev
                and isinstance(dev, str)
                and dev.strip().lower() not in ("null", "none", "")
                and len(dev) <= 20          # 只保留短标签，过滤掉 LLM 生成的长叙述句
                and dev not in state.personality_traits
            ):
                state.personality_traits.append(dev)

        # 检查是否需要压缩
        if self.enable_compression:
            for name, state in self.characters.items():
                if len(state.known_facts) > self.compress_threshold:
                    self._compress_state(name)

    def _update_relationship(self, state: CharacterState, rel_change: dict):
        target = rel_change.get("target", "")
        if not target:
            return
        # 查找已有关系
        for rel in state.relationships:
            if rel.target == target:
                rel.description = rel_change.get("change", rel.description)
                return
        # 新建关系
        state.relationships.append(RelationshipState(
            target=target,
            relation_type=rel_change.get("type", "相关"),
            description=rel_change.get("change", ""),
            trust_level=rel_change.get("trust_level", "中"),
        ))

    def _compress_state(self, name: str):
        """压缩角色状态 — 这是记忆系统的核心算法"""
        state = self.characters[name]

        prompt = COMPRESSION_PROMPT.format(
            name=name,
            chapter=state.chapter,
            current_state=json.dumps({
                "identity": state.identity,
                "personality_traits": state.personality_traits,
                "speech_style": state.speech_style,
                "known_facts": state.known_facts,
                "unknown_facts": state.unknown_facts,
                "emotional_state": state.emotional_state,
                "goals": state.goals,
                "key_memories": state.key_memories,
                "relationships": [
                    {"target": r.target, "relation_type": r.relation_type,
                     "description": r.description, "trust_level": r.trust_level}
                    for r in state.relationships
                ],
            }, ensure_ascii=False, indent=2),
        )

        compressed = self.llm.generate_json(prompt)

        # 用压缩结果更新状态
        state.identity = compressed.get("identity", state.identity)
        state.personality_traits = compressed.get("personality_traits", state.personality_traits)
        state.speech_style = compressed.get("speech_style", state.speech_style)
        state.known_facts = compressed.get("known_facts", state.known_facts)
        state.unknown_facts = compressed.get("unknown_facts", state.unknown_facts)
        state.emotional_state = compressed.get("emotional_state", state.emotional_state)
        state.goals = compressed.get("goals", state.goals)
        state.key_memories = compressed.get("key_memories", state.key_memories)

        # 重建关系
        state.relationships = [
            RelationshipState(
                target=r["target"],
                relation_type=r.get("relation_type", "相关"),
                description=r.get("description", ""),
                trust_level=r.get("trust_level", "中"),
            )
            for r in compressed.get("relationships", [])
        ]

    def get_state(self, name: str, at_chapter: int | None = None) -> CharacterState | None:
        """获取角色状态。如果指定 at_chapter，从历史重建（暂时返回当前状态）"""
        return self.characters.get(name)

    def get_state_at_chapter(self, name: str, target_chapter: int) -> CharacterState | None:
        """
        获取角色在指定章节的状态快照。

        策略：从头重新回放到 target_chapter。
        （MVP阶段简单实现，后续可优化为checkpoint机制）
        """
        if name not in self.characters:
            return None

        # 创建一个临时 manager 重建到指定章节
        temp_manager = CharacterStateManager(
            self.llm,
            self.compress_threshold,
            enable_compression=False,
        )
        for record in self._history:
            if record["chapter"] > target_chapter:
                break
            temp_manager.update_from_chapter(record["chapter"], record["extraction"])

        return temp_manager.characters.get(name)

    def list_characters(self) -> list[str]:
        return list(self.characters.keys())

    def save(self, path: str | Path):
        """保存到 JSON 文件"""
        data = {
            "characters": {},
            "history": self._history,
        }
        for name, state in self.characters.items():
            data["characters"][name] = {
                "name": state.name,
                "chapter": state.chapter,
                "identity": state.identity,
                "personality_traits": state.personality_traits,
                "speech_style": state.speech_style,
                "known_facts": state.known_facts,
                "unknown_facts": state.unknown_facts,
                "emotional_state": state.emotional_state,
                "goals": state.goals,
                "key_memories": state.key_memories,
                "relationships": [
                    {"target": r.target, "relation_type": r.relation_type,
                     "description": r.description, "trust_level": r.trust_level}
                    for r in state.relationships
                ],
            }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path, llm: LLMClient) -> "CharacterStateManager":
        """从 JSON 文件加载"""
        data = json.loads(Path(path).read_text())
        manager = cls(llm)
        manager._history = data.get("history", [])

        for name, state_data in data.get("characters", {}).items():
            manager.characters[name] = CharacterState(
                name=state_data["name"],
                chapter=state_data["chapter"],
                identity=state_data["identity"],
                personality_traits=state_data["personality_traits"],
                speech_style=state_data["speech_style"],
                known_facts=state_data["known_facts"],
                unknown_facts=state_data["unknown_facts"],
                emotional_state=state_data["emotional_state"],
                goals=state_data["goals"],
                key_memories=state_data.get("key_memories", []),
                relationships=[
                    RelationshipState(**r) for r in state_data.get("relationships", [])
                ],
            )
        return manager
