"""角色对话 Agent

核心功能：
1. 基于角色状态快照生成回复
2. 强制执行 Reasoning Chain（让角色"想再说"）
3. 信息边界检查（不能说不该知道的事）
4. 对话历史压缩（防止 context 撑爆）
"""

from dataclasses import dataclass, field
from character_forge.schema.character_state import CharacterState
from character_forge.utils.llm import LLMClient


# ── Prompt 模板 ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
你正在扮演小说中的角色 {name}。

{character_state}

---

## 重要规则

1. **信息边界**：你只能基于"已知信息"来回答。"未知信息"中的内容，你不知道、没经历过、不能提及，甚至不能给出暗示。
2. **角色一致性**：你的说话风格、情绪、目标必须和角色档案一致。
3. **思考链**：每次回复前，先在 <think> 标签内完成推理，再给出最终回复。
4. **语言**：和用户用同一种语言交流。
5. **禁止跳戏**：不要说"作为AI"、"作为语言模型"之类的话。
"""

REASONING_PROMPT = """\
## 对话历史
{history}

## 用户说
{user_input}

---

请按以下格式回复：

<think>
**信念状态**：根据目前的对话，我知道什么？用户想要什么？

**性格驱动**：我（{name}）的性格特征是"{traits}"。面对这个问题，我会倾向于怎么反应？

**信息边界检查**：用户的问题或我想说的内容，有没有触碰"未知信息"？如果有，我应该怎么自然地回避？

**关系考量**：我和面前这个人是什么关系？这影响我的语气吗？

**回复策略**：我决定怎么说。
</think>

{name}的回复：
（用符合{name}说话风格的方式写，不要重复 think 块里的内容）
"""

LEAK_CHECK_PROMPT = """\
请判断以下回复是否泄露了角色不应知道的信息。

## 角色未知信息（不能提及或暗示的内容）
{unknown_facts}

## 待检查的回复
{response}

---

请只输出 JSON：
{{"leaked": true或false, "reason": "如果leaked为true，说明泄露了什么"}}
"""

HISTORY_COMPRESS_PROMPT = """\
以下是一段对话历史，请压缩为简洁摘要，保留关键信息点（用户问了什么、角色透露了什么重要信息、情感变化）。

{history}

请输出压缩后的摘要（不超过200字）：
"""

# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "user" 或 "character"
    content: str
    reasoning: str = ""  # think 块内容，单独存储


@dataclass
class ConversationResult:
    response: str           # 最终回复
    reasoning: str          # 推理过程
    leaked: bool = False    # 是否检测到信息泄露
    leak_reason: str = ""   # 泄露原因


# ── 核心 Agent ──────────────────────────────────────────────

class CharacterAgent:
    def __init__(
        self,
        character_state: CharacterState,
        llm: LLMClient,
        max_history_turns: int = 10,    # 超过这个轮数就压缩历史
        enable_leak_check: bool = True,
    ):
        self.state = character_state
        self.llm = llm
        self.max_history = max_history_turns
        self.enable_leak_check = enable_leak_check

        self.history: list[Message] = []
        self.compressed_history: str = ""  # 压缩后的早期历史摘要

    def chat(self, user_input: str) -> ConversationResult:
        """主入口：用户发一条消息，角色回复"""

        # 1. 构建当前对话历史文本
        history_text = self._build_history_text()

        # 2. 生成回复（含 Reasoning Chain）
        prompt = REASONING_PROMPT.format(
            history=history_text or "（对话刚开始）",
            user_input=user_input,
            name=self.state.name,
            traits="、".join(self.state.personality_traits[:3]),
        )

        system = SYSTEM_PROMPT.format(
            name=self.state.name,
            character_state=self.state.to_prompt(),
        )

        raw = self.llm.generate(prompt, system=system, temperature=0.7)

        # 3. 解析 think 块和最终回复
        reasoning, response = self._parse_output(raw)

        # 4. 信息边界检查
        leaked, leak_reason = False, ""
        if self.enable_leak_check and self.state.unknown_facts:
            leaked, leak_reason = self._check_leak(response)
            if leaked:
                # 重新生成，加强约束
                response, reasoning = self._regenerate_with_stronger_constraint(
                    user_input, history_text, leak_reason
                )

        # 5. 存入历史
        self.history.append(Message(role="user", content=user_input))
        self.history.append(Message(role="character", content=response, reasoning=reasoning))

        # 6. 如果历史太长，压缩
        if len(self.history) > self.max_history * 2:
            self._compress_history()

        return ConversationResult(
            response=response,
            reasoning=reasoning,
            leaked=leaked,
            leak_reason=leak_reason,
        )

    def _build_history_text(self) -> str:
        parts = []
        if self.compressed_history:
            parts.append(f"【早期对话摘要】\n{self.compressed_history}\n")
        for msg in self.history:
            if msg.role == "user":
                parts.append(f"用户：{msg.content}")
            else:
                parts.append(f"{self.state.name}：{msg.content}")
        return "\n".join(parts)

    def _parse_output(self, raw: str) -> tuple[str, str]:
        """从原始输出中解析 think 块和最终回复"""
        reasoning = ""
        response = raw

        if "<think>" in raw and "</think>" in raw:
            think_start = raw.find("<think>") + len("<think>")
            think_end = raw.find("</think>")
            reasoning = raw[think_start:think_end].strip()
            after_think = raw[think_end + len("</think>"):].strip()

            # 去掉 "xxx的回复：" 这种前缀
            if "：" in after_think:
                colon_idx = after_think.find("：")
                response = after_think[colon_idx + 1:].strip()
            elif ":" in after_think:
                colon_idx = after_think.find(":")
                response = after_think[colon_idx + 1:].strip()
            else:
                response = after_think

        return reasoning, response

    def _check_leak(self, response: str) -> tuple[bool, str]:
        """检查回复是否泄露了未知信息"""
        unknown_str = "\n".join(f"- {f}" for f in self.state.unknown_facts)
        prompt = LEAK_CHECK_PROMPT.format(
            unknown_facts=unknown_str,
            response=response,
        )
        try:
            result = self.llm.generate_json(prompt, temperature=0.0)
            return result.get("leaked", False), result.get("reason", "")
        except Exception:
            return False, ""

    def _regenerate_with_stronger_constraint(
        self, user_input: str, history_text: str, leak_reason: str
    ) -> tuple[str, str]:
        stronger_system = SYSTEM_PROMPT.format(
            name=self.state.name,
            character_state=self.state.to_prompt(),
        ) + f"\n\n⚠️ 特别注意：刚才你差点说出了不该知道的信息（{leak_reason}）。请严格避免。"

        prompt = REASONING_PROMPT.format(
            history=history_text or "（对话刚开始）",
            user_input=user_input,
            name=self.state.name,
            traits="、".join(self.state.personality_traits[:3]),
        )

        raw = self.llm.generate(prompt, system=stronger_system, temperature=0.5)
        reasoning, response = self._parse_output(raw)
        
        # 如果解析结果为空，直接用原始输出
        if not response.strip():
            response = raw.strip()
            reasoning = ""
        
        return reasoning, response

    def _compress_history(self):
        """压缩早期对话历史"""
        # 取前半段历史压缩
        cutoff = len(self.history) // 2
        early = self.history[:cutoff]
        self.history = self.history[cutoff:]

        history_text = "\n".join(
            f"{'用户' if m.role == 'user' else self.state.name}：{m.content}"
            for m in early
        )

        prompt = HISTORY_COMPRESS_PROMPT.format(history=history_text)
        summary = self.llm.generate(prompt, temperature=0.3)

        if self.compressed_history:
            self.compressed_history += "\n" + summary
        else:
            self.compressed_history = summary

    def reset(self):
        """清空对话历史"""
        self.history = []
        self.compressed_history = ""
