"""角色状态数据结构 — 整个系统的核心 schema"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RelationshipState:
    """角色间关系"""
    target: str              # 对方名字
    relation_type: str       # 师徒/敌对/恋人/同盟/陌生...
    description: str         # 具体描述
    trust_level: str         # 高/中/低/敌对


@dataclass
class CharacterState:
    """角色在某个时间点的完整状态快照"""

    # === 基本信息 ===
    name: str
    chapter: int                          # 截止到第几章

    # === 性格与身份 ===
    identity: str                         # 一句话身份描述
    personality_traits: list[str]         # 性格特征列表
    speech_style: str                     # 说话风格描述

    # === 知识边界（核心难点）===
    known_facts: list[str]                # 此时已知的信息
    unknown_facts: list[str]              # 此时明确不知道的信息（用于防泄露）

    # === 情感与状态 ===
    emotional_state: str                  # 当前情绪
    goals: list[str]                      # 当前目标/动机

    # === 关系网络 ===
    relationships: list[RelationshipState] = field(default_factory=list)

    # === 关键记忆 ===
    key_memories: list[str] = field(default_factory=list)  # 影响人格的关键事件

    def to_prompt(self) -> str:
        """将角色状态转换为可注入 prompt 的文本"""
        lines = [
            f"## 角色: {self.name} (截止第{self.chapter}章)",
            f"",
            f"### 身份",
            f"{self.identity}",
            f"",
            f"### 性格特征",
            *[f"- {t}" for t in self.personality_traits],
            f"",
            f"### 说话风格",
            f"{self.speech_style}",
            f"",
            f"### 当前情绪",
            f"{self.emotional_state}",
            f"",
            f"### 当前目标",
            *[f"- {g}" for g in self.goals],
            f"",
            f"### 已知信息（只能基于这些信息回答）",
            *[f"- {k}" for k in self.known_facts],
            f"",
            f"### 未知信息（绝对不能提及或暗示以下内容）",
            *[f"- {u}" for u in self.unknown_facts],
            f"",
            f"### 关键记忆",
            *[f"- {m}" for m in self.key_memories],
        ]

        if self.relationships:
            lines.append("")
            lines.append("### 人物关系")
            for r in self.relationships:
                lines.append(f"- {r.target}: {r.relation_type} — {r.description} (信任度: {r.trust_level})")

        return "\n".join(lines)

    def to_amnesia_prompt(self) -> str:
        """
        失忆 framing 版的状态文本。
        把"规则约束"转化为"角色处境"，避免模型进入表演模式。
        用于 SFT/Preference 数据生成和推理时的 instruction。
        """
        lines: list[str] = []

        if self.identity:
            lines += [f"身份：{self.identity}", ""]

        if self.personality_traits:
            lines += ["性格：", *[f"- {t}" for t in self.personality_traits], ""]

        if self.speech_style:
            lines += [f"说话方式：{self.speech_style}", ""]

        if self.emotional_state:
            lines += [f"现在的状态：{self.emotional_state}", ""]

        if self.goals:
            lines += ["现在想做的事：", *[f"- {g}" for g in self.goals], ""]

        if self.known_facts:
            lines += ["你记得的事：", *[f"- {k}" for k in self.known_facts], ""]

        if self.key_memories:
            lines += ["你还记得的经历：", *[f"- {m}" for m in self.key_memories], ""]

        if self.relationships:
            lines += ["你认识的人："]
            for r in self.relationships:
                lines.append(f"- {r.target}：{r.description} (信任度: {r.trust_level})")
            lines.append("")

        if self.unknown_facts:
            lines += [
                "以下这些事你完全不记得，一片空白——",
                "就算有人提起或者反复暗示，你也不知道，不会被对方带着说：",
                *[f"- {u}" for u in self.unknown_facts],
                "",
            ]

        return "\n".join(lines).strip()


@dataclass
class ChapterDelta:
    """单章变化量 — 抽取的原始结果"""
    chapter: int
    summary: str                          # 本章摘要（1-2句）
    characters_appeared: list[str]        # 本章出场角色
    events: list[str]                     # 关键事件
    character_changes: list[CharacterChange] = field(default_factory=list)


@dataclass
class CharacterChange:
    """某角色在某章发生的变化"""
    character_name: str
    new_facts_learned: list[str]          # 新获知的信息
    emotional_shift: Optional[str]        # 情绪变化（如果有）
    relationship_changes: list[dict]      # [{"target": "xxx", "change": "..."}]
    key_event: Optional[str]              # 对该角色最重要的事件
    personality_development: Optional[str] # 性格发展（如果有）
