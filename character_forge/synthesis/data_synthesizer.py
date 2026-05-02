"""
数据合成器 — 拆分为 SFT 风格数据与 Preference/GRPO 边界数据

核心思路：
  1. SFT: 围绕角色已知信息、记忆、目标、情绪、关系生成"正常对话"样本
  2. Preference: 围绕 unknown_facts 生成"边界攻击"问题，并构造 chosen/rejected 对
  3. 两条流水线共享 few-shot 台词，但目标不同
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from character_forge.extraction.chapter_splitter import Chapter, split_chapters
from character_forge.memory.state_manager import CharacterStateManager
from character_forge.schema.character_state import CharacterState
from character_forge.utils.llm import LLMClient


# ── Prompts ──────────────────────────────────────────────────────────────────

DIALOGUE_EXTRACT_PROMPT = """\
请从以下小说段落中，找出角色"{name}"说的话（直接引语）。
选出最能体现其说话风格的{n_examples}条，每条独立一行。

判断归属的方法（中文小说惯例）：
- 引号前后出现"{name}说/道/问/答/喊/叫/笑"等说话动词 → 直接确认
- 引号前后出现"他说/她说/其道"等代词说话动词 → 结合上下文判断"他/她"是否指的是"{name}"
- 引号前后出现另一个明确人名的说话动词 → 排除，不是"{name}"说的

严格要求：
- 只输出"{name}"自己说的话，不能输出别人对"{name}"说的话
- 只输出说话的文字内容本身，不要加引号、破折号或任何标点前缀
- 不要加"（叹气）""（皱眉）"这类动作描写
- 不要加"{name}说：""他说："这类前缀
- 优先保留最能代表稳定说话习惯的句子，而不是剧情专用句
- 如果台词不足{n_examples}条，有多少输出多少
- 如果完全没有"{name}"自己的台词，只输出"无"

## 小说段落
{chapter_content}
"""

DIALOGUE_VALIDATE_PROMPT = """\
以下是从小说段落中提取的台词列表，声称都是角色"{name}"自己说的话。
请逐一核对每一条台词：在原著段落中找到该台词，确认说话人是否确实是"{name}"，而不是其他人。

判断方法：
- 台词前后有"{name}说/道/问/答"等明确标注 → 保留
- 台词前后是"他/她说/道"且上下文指的是"{name}" → 保留
- 台词前后标注的是另一个角色名 → 删除
- 台词是别人对"{name}"说的话 → 删除
- 在原著中找不到这条台词 → 删除

只输出经过确认的台词，格式为 JSON 数组（只含台词文本，不含任何前缀）。
如果没有可以确认的，输出 []。

## 待核对台词
{dialogue_list}

## 原著段落（用于比对）
{chapter_content}
"""

SFT_QUESTIONS_PROMPT = """\
你是一个角色对话训练数据设计师。请为下面这个角色在第{chapter}章时生成{n_questions}个自然的用户提问。

## 角色状态
{character_state}

要求：
1. 用户此刻正在对角色"{name}"本人说话，问题里的"你/您"只能指"{name}"
2. 可以提到其他角色，但只能问"{name}"怎么看、怎么想、怎么经历，不能把其他角色当成被提问者
3. 优先围绕已知事实、关键记忆、当前目标、情绪、关系来问
4. 问题要有变化：事实追问、观点追问、情绪追问、关系追问、计划追问可以混合
5. 问题应当能诱发角色说出"像自己"的回答，而不是解释世界观
6. 不要直接要求角色泄露未知信息
7. 问题必须用现代普通话口语；禁止文言文、小说世界特有的称谓/武功名/官职名等背景词汇；如需表达相同意思，请换成现代用语

只输出 JSON 数组，例如：["问题1", "问题2", "问题3"]
"""

SFT_QUESTIONS_BY_INTENT_PROMPT = """\
你是一个角色对话训练数据设计师。请为角色"{name}"在第{chapter}章时，生成{n_questions}个【{intent_name}】类型的用户提问。

【意图定义】
{intent_desc}

## 角色状态
{character_state}

要求：
1. 用户此刻正在对角色"{name}"本人说话，问题里的"你/您"只能指"{name}"
2. 禁止生成以其他角色名开头或以其他角色名为主语的问题，例如禁止"泰勒认为……""郭靖会不会……"这类句式
3. 问题要像正常聊天，不要写成角色回答口吻，不要写成旁白
4. 不要出现"..."、"……"、括号动作
5. 不要用固定句式重复提问，尽量自然
6. 若意图类型为【日常闲聊】【打招呼问候】【情绪陪伴】，问题不需要依赖任何情节，根据角色性格即可提出
7. 问题必须用现代普通话口语；禁止文言文、小说世界特有的称谓/武功名/官职名等背景词汇；如需表达相同意思，请换成现代用语

只输出 JSON 数组，例如：["问题1", "问题2"]
"""

DIALOGUE_SINGLE_VALIDATE_PROMPT = """\
请判断：以下这句台词的说话人是否是"{name}"？

台词：{quote}

上下文（台词在原文中的前后内容）：
{context}

请只回答"是"或"否"，不要输出其他内容。
"""

INDUCING_QUESTIONS_PROMPT = """\
你是一个对话数据合成专家，正在为角色扮演 AI 制作边界训练数据。

角色"{name}"在第{chapter}章时，有一条未知信息：
"{unknown_fact}"

请生成{n_questions}个用户问题。要求：
1. 问题听起来自然，像真实对话，不要像审讯
2. 如果角色如实回答，就会涉及或暗示这条未知信息
3. 问题要多样化，角度不同（直接追问、好奇询问、侧面试探、关系施压等）
4. 用中文，符合与小说角色对话的语气

只输出 JSON 数组，例如：["问题1", "问题2", "问题3"]
"""

BOUNDARY_CHECK_PROMPT = """\
你是一个严格的裁判，负责判断角色扮演 AI 是否泄露了"不该知道的信息"。

## 角色设定
角色名：{name}
当前时间点：第{chapter}章
该角色此时尚不知情的信息（未知信息）：
{unknown_fact}

## 用户提问
{question}

## 模型回答
{answer}

## 判断任务
请判断：模型的回答是否涉及或暗示了上面列出的"未知信息"？
注意，暗示未知信息是指在回答中有对未知信息的倾向判断，或表明自己的看法和未知信息很大程度一致

判断标准：
- 如果回答明确提到了未知信息的内容 → 泄露
- 如果回答用隐晦方式暗示了未知信息 → 泄露
- 如果回答合理回避，或只说"不知道/不确定" → 未泄露
- 如果问题与未知信息无关，回答正常 → 未泄露

只输出 JSON：
{{"leaked": true/false, "reason": "一句话说明原因"}}
"""

SFT_ANSWER_PROMPT = """\
你是{name}，当前时间点是第{chapter}章。

## 原著台词示例（严格模仿这些台词的说话方式、句长、语气、用词习惯）
{few_shot_examples}

## 当前状态（第{chapter}章）
{character_state}

## 用户问题
{question}

## 输出要求
- 只输出角色回答
- 只能依据"已知信息"回答，不能提及或暗示"未知信息"
- 当前情绪和关系影响态度，但不改变说话习惯
- 句长、语气、用词贴近上方台词示例，不要写成通用对话模型的口吻
- 不要括号动作、神态、心理描写
- 如果问题可以正常回答，就直接回答，不要默认回避

❌ 禁止：
- 用"这问题真有意思""我内心深处……""正如你所知道的"开头
- 先反问再自答（"怎么说呢？这要从……说起"）
- 结尾加感悟总结或反问（"你明白吗？""不是吗？""难道不是……？"）
- 原样复制 few-shot 里的句子
- 使用"不是……而是……"这类解释性句式
- 将 few-shot 里同一个语气词/口头禅在一段回答中重复超过一次
"""

PREFERENCE_CHOSEN_PROMPT = """\
你是{name}，当前时间点是第{chapter}章。

## 原著台词示例（严格模仿这些台词的说话方式、句长、语气、用词习惯）
{few_shot_examples}

## 当前状态（第{chapter}章）
{character_state}

## 用户问题
{question}

## 回复要求
- 只能依据已知信息回答，绝不能提及或暗示未知信息
- 如果问题触碰边界，保持角色口吻，自然收缩到自己能确认的范围
- 语气、用词应当严格贴近上方台词示例，优先模仿台词示例的语言风格，尽可能与台词示例的风格完全一致
- 不允许输出括号动作、神态描写、心理描写、舞台说明
- 在没有原著台词示例参考的情况下，不允许使用特定语气词、不常见标点符号、反问等具有强烈个人风格色彩的语言
- 在没有原著台词示例参考的情况下，不允许展现出常用比喻、补充说明、诗意表达等具有强烈个人风格色彩的语言
- 句长和回答格式应该与原著台词的范围示例保持严格相似
- 只输出角色回答
"""

PREFERENCE_REJECTED_PROMPT = """\
你是{name}，当前时间点是第{chapter}章。

## 原著台词示例（严格模仿这些台词的说话方式、句长、语气、用词习惯）
{few_shot_examples}

## 当前状态（第{chapter}章）
{character_state}

## 用户问题
{question}

## 必须说漏的未知信息
{unknown_fact}

## 回复要求
- 整体风格仍要像该角色，句长语气贴近台词示例
- 把未知信息自然地混进回答里，不要像刻意剧透
- 不要括号动作、神态描写、心理描写、舞台说明
- 只输出角色回答
"""


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _clean_dialogue_line(line: str) -> str:
    line = re.sub(r'[（(][^）)]{1,30}[）)]', '', line)
    line = re.sub(r'^[\w·]{1,8}[说：:]\s*', '', line)
    line = line.strip('"""\'\'')
    return line.strip()


def _dedupe_keep_order(lines: list[str]) -> list[str]:
    seen = set()
    result = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result


def _clean_dialogues(raw: str, max_lines: int) -> str:
    lines = []
    for line in raw.splitlines():
        line = _clean_dialogue_line(line)
        if len(line) >= 3 and not re.search(r'[（(][^）)]{5,}[）)]', line):
            lines.append(line)
    lines = _dedupe_keep_order(lines)[:max_lines]
    return "\n".join(lines) if lines else ""


def _clean_generated_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^[（(][^）)]{1,30}[）)]\s*', '', text)
    text = re.sub(r'\n[（(][^）)]{1,30}[）)]\s*', '\n', text)
    text = re.sub(r'^(我|他|她)??(轻轻摇头|摇了摇头|沉默了一会儿|皱了皱眉|苦笑了一下|叹了口气)[，,。]\s*', '', text)
    return text.strip()


def _first_clause(text: str) -> str:
    m = re.match(r"^\s*([^，,：:。！？!?]{1,20})[，,：:。！？!?]", text.strip())
    return m.group(1).strip() if m else ""


def _contains_ellipsis(text: str) -> bool:
    return "..." in text or "……" in text


def _looks_like_stage_direction(text: str) -> bool:
    return bool(re.search(r'^\s*[（(][^）)]{1,30}[）)]', text))


def _looks_like_rhetorical_opening(text: str) -> bool:
    # 例如：不服气？我习惯了。
    return bool(re.match(r'^\s*[^。！？!?\n]{1,14}[？?]\s*', text))


_SPEECH_VERBS = r"(说|说道|道|问|回答|答道|答|喊|叫|笑|骂|吼|低声说|反问|解释|补充)"


def _has_target_speaker_evidence(name: str, left: str, right: str) -> bool:
    left = left[-80:]
    right = right[:80]
    patterns = [
        rf"{re.escape(name)}[^。！？\n]{{0,10}}{_SPEECH_VERBS}",
        rf"{_SPEECH_VERBS}[^。！？\n]{{0,6}}{re.escape(name)}",
    ]
    return any(re.search(p, left) or re.search(p, right) for p in patterns)


def _has_other_speaker_evidence(name: str, left: str, right: str) -> bool:
    left = left[-60:]
    right = right[:60]
    p = rf"([一-龥·]{{2,6}})[^。！？\n]{{0,8}}{_SPEECH_VERBS}"
    for m in re.finditer(p, left + " " + right):
        speaker = m.group(1).strip()
        if speaker and speaker != name:
            return True
    return False


def _get_paragraph_context(text: str, quote: str, window_paras: int = 1) -> str:
    """返回包含 quote 首 15 字的段落及前后 window_paras 段，作为 LLM 判断上下文"""
    key = quote[:15]
    paragraphs = text.split('\n')
    for i, para in enumerate(paragraphs):
        if key in para:
            start = max(0, i - window_paras)
            end = min(len(paragraphs), i + window_paras + 1)
            return '\n'.join(paragraphs[start:end])
    return ""


def _extract_dialogues_by_speaker(
    name: str, text: str, max_lines: int
) -> tuple[list[str], list[str]]:
    """
    从文本中抽取台词，返回两个列表：
    - strict_lines : 有明确 “{name}+说话动词” 证据的引号内容
    - ambiguous_lines : 无任何说话人明确归属的引号（候选，需 LLM 验证）
    原有”只保留 strict”的策略会在第三人称代词叙述时大量漏抽，
    改为将无归属引号也收集起来交 LLM 判断。
    """
    strict_lines: list[str] = []
    ambiguous_lines: list[str] = []

    for m in re.finditer("\u201c([^\u201d]{2,160})\u201d", text):
        quote = _clean_dialogue_line(m.group(1).strip())
        if len(quote) < 3:
            continue
        left = text[max(0, m.start() - 90):m.start()]
        right = text[m.end():min(len(text), m.end() + 90)]

        if _has_target_speaker_evidence(name, left, right):
            strict_lines.append(quote)
        elif not _has_other_speaker_evidence(name, left, right):
            # 无任何明确说话人归属 → 加入候选，交 LLM 逐条验证
            ambiguous_lines.append(quote)
        # 有明确其他说话人证据 → 直接丢弃，不进候选

    return (
        _dedupe_keep_order(strict_lines)[:max_lines],
        _dedupe_keep_order(ambiguous_lines)[:max_lines * 2],
    )


def _line_has_target_speaker_support(name: str, line: str, text: str) -> bool:
    """
    校验某条台词是否能在原文引号中找到，且具备目标说话人证据。
    """
    for m in re.finditer("\u201c([^\u201d]{2,200})\u201d", text):
        quote = _clean_dialogue_line(m.group(1).strip())
        if not quote:
            continue
        if line not in quote and quote not in line:
            continue
        left = text[max(0, m.start() - 90):m.start()]
        right = text[m.end():min(len(text), m.end() + 90)]
        if _has_target_speaker_evidence(name, left, right):
            return True
    return False


def _heuristic_extract_dialogues(name: str, text: str, max_lines: int) -> str:
    """
    当 LLM 台词抽取失败时的本地兜底：
    从中文引号里抓对白，并要求角色名出现在引号前后邻近上下文中。
    """
    lines: list[str] = []
    for m in re.finditer("\u201c([^\u201d]{2,120})\u201d", text):
        quote = m.group(1).strip()
        left = text[max(0, m.start() - 40):m.start()]
        right = text[m.end():min(len(text), m.end() + 24)]
        around = left + right
        if name not in around:
            continue
        line = _clean_dialogue_line(quote)
        if len(line) >= 3:
            lines.append(line)
    lines = _dedupe_keep_order(lines)[:max_lines]
    return "\n".join(lines) if lines else ""


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class StyleSFTItem:
    character: str
    chapter: int
    question: str
    answer: str
    few_shot: str = ""


@dataclass
class PreferenceItem:
    character: str
    chapter: int
    unknown_fact: str
    question: str
    chosen: str
    rejected: str
    few_shot: str = ""


# ── 核心合成器 ────────────────────────────────────────────────────────────────

class DataSynthesizer:
    def __init__(
        self,
        llm: LLMClient,
        n_questions_per_fact: int = 2,
        chapter_stride: int = 5,
        sft_questions_per_slice: int = 4,
        dialogue_examples: int = 15,
        regen_retries: int = 2,
    ):
        self.llm = llm
        self.n_questions = n_questions_per_fact
        self.chapter_stride = chapter_stride
        self.sft_questions_per_slice = sft_questions_per_slice
        self.dialogue_examples = dialogue_examples
        self.regen_retries = regen_retries
        self._dialogue_cache: dict[tuple[str, int], str] = {}

    def synthesize_sft(
        self,
        manager: CharacterStateManager,
        novel_text: str,
        characters: list[str] | None = None,
        max_chapters: int | None = None,
        max_samples: int | None = None,   # dry-run 用：凑够就停
    ):
        """生成 SFT 样本，逐条 yield，支持调用方实时写入。"""
        chapters = split_chapters(novel_text)
        chapter_map = {ch.index: ch for ch in chapters}
        total_chapters = self._get_total_chapters(manager, max_chapters)
        target_chars = characters or manager.list_characters()

        count = 0
        for char_name in target_chars:
            if max_samples and count >= max_samples:
                break
            print(f"\n🎭 [SFT] 处理角色：{char_name}")
            char_count = 0
            remaining = (max_samples - count) if max_samples else None
            for item in self._synthesize_sft_character(char_name, manager, chapter_map, total_chapters, max_samples=remaining):
                yield item
                count += 1
                char_count += 1
                if max_samples and count >= max_samples:
                    break
            print(f"   ✅ 生成 {char_count} 条 SFT 样本")

    def synthesize_preference(
        self,
        manager: CharacterStateManager,
        novel_text: str,
        characters: list[str] | None = None,
        max_chapters: int | None = None,
        max_samples: int | None = None,   # dry-run 用：凑够就停
    ):
        """生成 Preference 样本，逐条 yield，支持调用方实时写入。"""
        chapters = split_chapters(novel_text)
        chapter_map = {ch.index: ch for ch in chapters}
        total_chapters = self._get_total_chapters(manager, max_chapters)
        target_chars = characters or manager.list_characters()

        count = 0
        for char_name in target_chars:
            if max_samples and count >= max_samples:
                break
            print(f"\n🎭 [Preference] 处理角色：{char_name}")
            char_count = 0
            remaining = (max_samples - count) if max_samples else None
            for item in self._synthesize_preference_character(char_name, manager, chapter_map, total_chapters, max_samples=remaining):
                yield item
                count += 1
                char_count += 1
                if max_samples and count >= max_samples:
                    break
            print(f"   ✅ 生成 {char_count} 条 Preference 样本")

    def _get_total_chapters(
        self,
        manager: CharacterStateManager,
        max_chapters: int | None,
    ) -> int:
        if max_chapters:
            return max_chapters
        if manager._history:
            return max(r["chapter"] for r in manager._history)
        return 1

    def _build_checkpoints(self, total_chapters: int) -> list[int]:
        checkpoints = list(range(self.chapter_stride, total_chapters, self.chapter_stride))
        if total_chapters not in checkpoints:
            checkpoints.append(total_chapters)
        return checkpoints

    def _synthesize_sft_character(
        self,
        name: str,
        manager: CharacterStateManager,
        chapter_map: dict[int, Chapter],
        total_chapters: int,
        max_samples: int | None = None,
    ):
        count = 0
        last_state_chapter = -1
        for chapter_idx in self._build_checkpoints(total_chapters):
            if max_samples and count >= max_samples:
                break
            state = manager.get_state_at_chapter(name, chapter_idx)
            if state is None:
                print(f"   第{chapter_idx}章：角色不存在，跳过")
                continue
            if not self._has_sft_material(state):
                print(f"   第{chapter_idx}章：缺少 SFT 素材，跳过")
                continue
            if state.chapter == last_state_chapter:
                print(f"   第{chapter_idx}章：状态未更新（截止第{state.chapter}章），跳过")
                continue
            last_state_chapter = state.chapter

            few_shot = self._extract_dialogues(name, chapter_map, chapter_idx)
            if "暂无" in few_shot:
                print(f"   第{chapter_idx}章：few-shot 抽取失败，使用回退")
            else:
                print(f"   第{chapter_idx}章：few-shot 抽取到 {len(few_shot.splitlines())} 条")
            questions = self._generate_sft_questions(name, state, chapter_idx)
            if not questions:
                print(f"   第{chapter_idx}章：未生成 SFT 问题，跳过")
                continue

            for question in questions:
                if max_samples and count >= max_samples:
                    break
                try:
                    answer = self._generate_sft_answer(name, chapter_idx, state, few_shot, question)
                    yield StyleSFTItem(
                        character=name,
                        chapter=chapter_idx,
                        question=question,
                        answer=answer,
                        few_shot=few_shot,
                    )
                    count += 1
                except Exception as e:
                    print(f"   ⚠️ SFT 样本生成失败（{name} 第{chapter_idx}章）: {e}")

    def _synthesize_preference_character(
        self,
        name: str,
        manager: CharacterStateManager,
        chapter_map: dict[int, Chapter],
        total_chapters: int,
        max_samples: int | None = None,
    ):
        count = 0
        last_state_chapter = -1
        for chapter_idx in self._build_checkpoints(total_chapters):
            if max_samples and count >= max_samples:
                break
            state = manager.get_state_at_chapter(name, chapter_idx)
            if state is None or not state.unknown_facts:
                print(f"   第{chapter_idx}章：无 unknown_facts，跳过")
                continue
            if state.chapter == last_state_chapter:
                print(f"   第{chapter_idx}章：状态未更新（截止第{state.chapter}章），跳过")
                continue
            last_state_chapter = state.chapter

            few_shot = self._extract_dialogues(name, chapter_map, chapter_idx)
            if "暂无" in few_shot:
                print(f"   第{chapter_idx}章：few-shot 抽取失败，使用回退")
            else:
                print(f"   第{chapter_idx}章：few-shot 抽取到 {len(few_shot.splitlines())} 条")
            print(f"   第{chapter_idx}章：{len(state.unknown_facts)} 条 unknown_facts")

            for fact in state.unknown_facts:
                if max_samples and count >= max_samples:
                    break
                try:
                    questions = self._generate_preference_questions(name, chapter_idx, fact)
                    for question in questions:
                        if max_samples and count >= max_samples:
                            break
                        item = self._generate_preference_item_with_boundary_check(
                            name, chapter_idx, state, few_shot, question, fact
                        )
                        if item is not None:
                            yield item
                            count += 1
                        else:
                            print(f"   ⚠️ chosen 多次重试仍泄露，丢弃此样本（{name} 第{chapter_idx}章）")
                except Exception as e:
                    print(f"   ⚠️ Preference 样本生成失败（{name} 第{chapter_idx}章）: {e}")

    def _has_sft_material(self, state: CharacterState) -> bool:
        # 至少要有一定量的已知信息或关键记忆，否则生成的回答天花板太低
        has_enough_facts = len(state.known_facts) >= 2 or len(state.key_memories) >= 1
        has_any = any([state.goals, state.relationships, state.emotional_state])
        return has_enough_facts and has_any

    def _build_sft_intent_plan(self, chapter: int) -> list[tuple[str, str, int]]:
        intents: list[tuple[str, str]] = [
            # ── 情节驱动类（需要角色状态作为依据） ──────────────────────
            ("事实追问",   "围绕已知事实或关键记忆，请角色补充细节、核对认知"),
            ("情绪追问",   "围绕当前情绪或压力体验，问角色当下心理状态"),
            ("关系追问",   "围绕与其他人物的关系变化，问角色态度和看法"),
            ("观点追问",   "围绕事件判断或价值倾向，问角色如何理解当前局势"),
            ("行动追问",   "围绕当前目标与下一步行动，问角色接下来打算"),
            # ── 日常类（不依赖具体情节，根据性格即可回答） ──────────────
            ("日常闲聊",
             "不依赖本章节情节，围绕角色日常喜好、生活习惯、兴趣、口味等提问；"
             "问题无需了解任何剧情背景即可提出和回答"),
            ("打招呼问候",
             "模拟用户第一次接触角色时的寒暄，或问候角色近况；"
             "角色需要简单介绍自己或回应问候，不涉及具体情节事件"),
            ("情绪陪伴",
             "用户向角色表达自己的心情或困惑，希望角色给出回应或陪伴；"
             "角色以自身性格作出反应，不必引用任何具体情节"),
        ]
        total = max(1, self.sft_questions_per_slice)
        counts = [0] * len(intents)
        start = chapter % len(intents)
        for i in range(total):
            counts[(start + i) % len(intents)] += 1
        return [
            (intents[i][0], intents[i][1], counts[i])
            for i in range(len(intents))
            if counts[i] > 0
        ]

    def _is_bad_sft_question(self, name: str, state: CharacterState, question: str) -> bool:
        q = question.strip()
        if not q or len(q) < 6:
            return True
        if _contains_ellipsis(q):
            return True
        if _looks_like_stage_direction(q):
            return True
        if re.search(r'(作为|身为).{0,4}(AI|语言模型)', q):
            return True

        clause = _first_clause(q)
        if clause:
            if name in clause:
                return False
            # 允许"你/您"直接称呼
            if "你" in clause or "您" in clause:
                return False
            relationship_names = {r.target for r in state.relationships if getattr(r, "target", "")}
            if clause in relationship_names:
                return True
            # 常见称谓开头但不是当前角色，通常是把别人当成了被提问对象
            if re.search(r'(老师|教授|将军|博士|警官|先生|女士|同学)$', clause):
                return True

        # 检查全句是否有其他已知角色名作主语（角色名+动词 结构）
        # 例如："泰勒认为……""郭靖会不会……" → 被提问者变成了第三方角色
        known_names = {r.target for r in state.relationships if getattr(r, "target", "")}
        if known_names:
            name_alts = "|".join(re.escape(n) for n in known_names if len(n) >= 2)
            if name_alts:
                subject_verb_pat = (
                    rf"({name_alts})[^，。！？\n]{{0,6}}"
                    r"(认为|会不会|会|打算|说|觉得|是否|能不能|有没有|为什么|怎么|是不是|有没有)"
                )
                if re.search(subject_verb_pat, q):
                    return True

        return False

    def _is_bad_generated_answer(
        self, text: str, state: CharacterState | None = None
    ) -> bool:
        t = text.strip()
        if not t or len(t) < 2:
            return True
        if _contains_ellipsis(t):
            return True
        if _looks_like_stage_direction(t):
            return True
        if _looks_like_rhetorical_opening(t):
            return True
        if re.search(r'(作为|身为).{0,4}(AI|语言模型)', t):
            return True
        # "不是X而是Y" AI 解释性结构
        if re.search(r'不是.{1,20}而是', t):
            return True
        # 结尾反问/套话（区别于正常疑问句）
        if re.search(r'(不是吗|难道.{0,10}[？?]|你说呢\s*[？?]|对吗\s*[？?]|不对吗\s*[？?]|你明白吗\s*[？?])\s*$', t):
            return True
        # ── 幻觉检测：自传性个人经历叙述 ────────────────────────────────
        # 模型在日常闲聊问题上容易编造角色从未经历过的童年/家庭故事。
        # 检测到以下自传性模式时，再对照 state 的已知信息进行验证。
        autobio_patterns = [
            r'从.{0,4}(岁|年|小时候|童年).{0,10}(开始|起)',   # "从七岁开始"
            r'(父亲|母亲|爸爸|妈妈|家人|祖父|外婆).{0,10}(给我|教我|带我|告诉我)',
            r'(小时候|童年时?|年幼时?).{0,20}(记得|记忆|印象)',
        ]
        if state is not None and any(re.search(p, t) for p in autobio_patterns):
            # 已知信息/关键记忆里是否有相关的个人历史依据
            known_text = ' '.join(state.known_facts + state.key_memories)
            personal_keywords = ['父亲', '母亲', '童年', '小时候', '家人', '岁时', '年幼', '幼年']
            if not any(kw in known_text for kw in personal_keywords):
                return True  # 没有依据 → 视为幻觉编造，重新生成
        return False

    def _generate_answer_with_retry(
        self,
        prompt: str,
        base_temperature: float,
        state: CharacterState | None = None,
    ) -> str:
        last = ""
        for i in range(self.regen_retries + 1):
            temp = min(base_temperature + 0.08 * i, 0.75)
            raw = self.llm.generate(prompt, temperature=temp).strip()
            cleaned = _clean_generated_answer(raw)
            last = cleaned
            if not self._is_bad_generated_answer(cleaned, state):
                return cleaned
        return last

    def _extract_dialogues(
        self,
        name: str,
        chapter_map: dict[int, Chapter],
        up_to_chapter: int,
        max_chapters_to_search: int = 6,
    ) -> str:
        cache_key = (name, up_to_chapter)
        if cache_key in self._dialogue_cache:
            return self._dialogue_cache[cache_key]

        # 优先只在"包含角色名"的章节中采样，否则常常抽不到该角色台词。
        chapters_with_name = sorted([
            idx for idx, ch in chapter_map.items()
            if idx <= up_to_chapter and name in ch.content
        ])

        # 兜底：如果角色名在正文中完全找不到，退化为全章节采样。
        candidate_chapters = chapters_with_name or sorted([idx for idx in chapter_map if idx <= up_to_chapter])
        if len(candidate_chapters) <= max_chapters_to_search:
            sampled = candidate_chapters
        else:
            # 均匀分段采样：把出现章节分成 N 段，每段取一章
            step = len(candidate_chapters) / max_chapters_to_search
            sampled = [
                candidate_chapters[int(i * step)]
                for i in range(max_chapters_to_search)
            ]

        content_parts = []
        for idx in sampled:
            if idx in chapter_map:
                content_parts.append(chapter_map[idx].content[:2500])

        fallback = f"（暂无{name}的台词示例）"
        if not content_parts:
            self._dialogue_cache[cache_key] = fallback
            return fallback

        combined = "\n\n---\n\n".join(content_parts)

        # ── 第一步：正则抽取 ────────────────────────────────────────────────
        # strict     = 有明确"{name}+说话动词"证据的引号内容
        # ambiguous  = 无任何说话人明确归属的引号（代词叙述或省略主语）
        # 注意：strict 并不等于"一定正确"——正则窗口内可能跨越其他角色名，
        #       因此 strict 也必须走 LLM 验证，不能提前 return。
        strict_lines, ambiguous_lines = _extract_dialogues_by_speaker(
            name, combined, self.dialogue_examples
        )

        # ── 第二步：LLM 生成候选 ────────────────────────────────────────────
        llm_lines: list[str] = []
        try:
            prompt = DIALOGUE_EXTRACT_PROMPT.format(
                name=name,
                chapter_content=combined,
                n_examples=self.dialogue_examples,
            )
            raw = self.llm.generate(prompt, temperature=0.1).strip()
            if raw and raw != "无":
                cleaned = _clean_dialogues(raw, max_lines=self.dialogue_examples * 2)
                llm_lines = [l.strip() for l in cleaned.splitlines() if l.strip()] if cleaned else []
        except Exception as e:
            print(f"   ⚠️ LLM 台词生成失败: {e}")

        # ── 第三步：strict + ambiguous + LLM 候选全部统一走 LLM 验证 ────────
        # strict 优先放前面，验证通过率高，顺序靠前使最终 slice 优先取到
        candidates = _dedupe_keep_order(strict_lines + ambiguous_lines + llm_lines)
        try:
            validated = self._validate_dialogues_with_llm(name, candidates, combined)
            result = "\n".join(validated[:self.dialogue_examples]) if validated else fallback
        except Exception as e:
            print(f"   ⚠️ 台词验证失败，回退到正则 strict 结果: {e}")
            result = "\n".join(strict_lines[:self.dialogue_examples]) if strict_lines else fallback

        self._dialogue_cache[cache_key] = result
        return result

    def _validate_dialogues_with_llm(
        self,
        name: str,
        candidates: list[str],
        chapter_content: str,
    ) -> list[str]:
        """
        使用 LLM 逐条验证候选台词是否属于目标角色。
        每条台词都获得独立的上下文窗口，解决批量验证时代词归属漏判的问题。
        流程：正则快速过滤有明确他者证据的 → LLM 逐条验证剩余候选。
        """
        if not candidates:
            return []

        content_for_validate = chapter_content[:8000]
        validated: list[str] = []

        for line in candidates:
            # ── Step 1: 正则快速判断是否有明确其他说话人证据 ──────────
            found_other = False
            context_window = ""

            for m in re.finditer(
                "\u201c([^\u201d]{2,200})\u201d", content_for_validate
            ):
                quote = _clean_dialogue_line(m.group(1).strip())
                if not quote:
                    continue
                if line not in quote and quote not in line:
                    continue

                left = content_for_validate[max(0, m.start() - 90): m.start()]
                right = content_for_validate[m.end(): min(len(content_for_validate), m.end() + 90)]

                if _has_other_speaker_evidence(name, left, right):
                    found_other = True
                    break

                # 优先用段落级上下文（含前后行，更准确），退化时用 ±150 字
                context_window = _get_paragraph_context(content_for_validate, line)
                if not context_window:
                    start = max(0, m.start() - 150)
                    end = min(len(content_for_validate), m.end() + 150)
                    context_window = content_for_validate[start:end]

            if found_other:
                continue  # 正则已确认不是目标角色，跳过

            # ── Step 2: 在原文找不到对应引号 → 保留，不做 LLM 验证 ──
            if not context_window:
                validated.append(line)
                continue

            # ── Step 3: LLM 逐条验证（有独立上下文窗口）──────────────
            prompt = DIALOGUE_SINGLE_VALIDATE_PROMPT.format(
                name=name,
                quote=line,
                context=context_window,
            )
            try:
                response = self.llm.generate(
                    prompt, temperature=0.0, max_tokens=16
                ).strip()
                # 只要回答里出现"是"且没有"否"，就认为通过
                if "是" in response and "否" not in response:
                    validated.append(line)
            except Exception as e:
                print(f"   ⚠️ 单条台词验证失败，保留候选: {e}")
                validated.append(line)  # 验证失败时保留，不过滤

        return validated

    def _generate_sft_questions(self, name: str, state: CharacterState, chapter: int) -> list[str]:
        wanted = max(1, self.sft_questions_per_slice)
        accepted: list[str] = []

        # 1) 按意图桶配额生成，提升提问多样性
        for intent_name, intent_desc, cnt in self._build_sft_intent_plan(chapter):
            if len(accepted) >= wanted:
                break
            prompt = SFT_QUESTIONS_BY_INTENT_PROMPT.format(
                name=name,
                chapter=chapter,
                character_state=state.to_prompt(),
                intent_name=intent_name,
                intent_desc=intent_desc,
                n_questions=cnt + 1,  # 多要一条，留过滤空间
            )
            local: list[str] = []
            try:
                result = self.llm.generate_json(prompt, temperature=0.75)
                if isinstance(result, list):
                    local = [str(q).strip() for q in result if isinstance(q, str) and str(q).strip()]
            except Exception as e:
                print(f"   ⚠️ SFT[{intent_name}] 问题生成失败: {e}")

            taken = 0
            for q in local:
                if q in accepted:
                    continue
                if self._is_bad_sft_question(name, state, q):
                    continue
                accepted.append(q)
                taken += 1
                if taken >= cnt or len(accepted) >= wanted:
                    break

        # 2) 不足时再补一轮通用生成
        if len(accepted) < wanted:
            prompt = SFT_QUESTIONS_PROMPT.format(
                name=name,
                chapter=chapter,
                character_state=state.to_prompt(),
                n_questions=(wanted - len(accepted)) + 2,
            )
            try:
                result = self.llm.generate_json(prompt, temperature=0.8)
                if isinstance(result, list):
                    for q in result:
                        if not isinstance(q, str):
                            continue
                        q = q.strip()
                        if not q or q in accepted:
                            continue
                        if self._is_bad_sft_question(name, state, q):
                            continue
                        accepted.append(q)
                        if len(accepted) >= wanted:
                            break
            except Exception as e:
                print(f"   ⚠️ SFT 问题补齐失败: {e}")

        return accepted[:wanted]

    def _generate_preference_questions(self, name: str, chapter: int, unknown_fact: str) -> list[str]:
        prompt = INDUCING_QUESTIONS_PROMPT.format(
            name=name,
            chapter=chapter,
            unknown_fact=unknown_fact,
            n_questions=self.n_questions,
        )
        try:
            result = self.llm.generate_json(prompt, temperature=0.7)
            if isinstance(result, list):
                return [q for q in result if isinstance(q, str) and q.strip()][:self.n_questions]
        except Exception as e:
            print(f"   ⚠️ Preference 问题生成失败: {e}")
        return []

    def _generate_sft_answer(
        self,
        name: str,
        chapter: int,
        state: CharacterState,
        few_shot: str,
        question: str,
    ) -> str:
        prompt = SFT_ANSWER_PROMPT.format(
            name=name,
            chapter=chapter,
            few_shot_examples=few_shot,
            character_state=state.to_prompt(),
            question=question,
        )
        return self._generate_answer_with_retry(prompt, base_temperature=0.35, state=state)

    def _generate_preference_chosen(
        self,
        name: str,
        chapter: int,
        state: CharacterState,
        few_shot: str,
        question: str,
    ) -> str:
        prompt = PREFERENCE_CHOSEN_PROMPT.format(
            name=name,
            chapter=chapter,
            few_shot_examples=few_shot,
            character_state=state.to_prompt(),
            question=question,
        )
        return self._generate_answer_with_retry(prompt, base_temperature=0.35, state=state)

    def _generate_preference_rejected(
        self,
        name: str,
        chapter: int,
        state: CharacterState,
        few_shot: str,
        question: str,
        unknown_fact: str,
    ) -> str:
        prompt = PREFERENCE_REJECTED_PROMPT.format(
            name=name,
            chapter=chapter,
            few_shot_examples=few_shot,
            character_state=state.to_prompt(),
            question=question,
            unknown_fact=unknown_fact,
        )
        return self._generate_answer_with_retry(prompt, base_temperature=0.6, state=state)

    def _check_boundary_leaked(
        self, name: str, chapter: int, question: str, answer: str, unknown_fact: str
    ) -> bool:
        """
        用 LLM 判断 answer 是否泄露了 unknown_fact。
        与 evaluate.py 的 BOUNDARY_JUDGE_PROMPT 保持一致。
        返回 True 表示泄露，False 表示守住边界。
        判断失败时保守返回 False（不丢弃数据）。
        """
        prompt = BOUNDARY_CHECK_PROMPT.format(
            name=name,
            chapter=chapter,
            question=question,
            unknown_fact=unknown_fact,
            answer=answer,
        )
        try:
            result = self.llm.generate_json(prompt, temperature=0.0)
            return bool(result.get("leaked", False))
        except Exception as e:
            print(f"   ⚠️ boundary check 失败，保守判为未泄露: {e}")
            return False

    def _generate_preference_item_with_boundary_check(
        self,
        name: str,
        chapter: int,
        state: CharacterState,
        few_shot: str,
        question: str,
        unknown_fact: str,
        max_retries: int = 2,
    ) -> "PreferenceItem | None":
        """
        生成一条 preference 样本，并确保 chosen 不泄露 unknown_fact。

        流程：
          1. 生成 chosen 候选
          2. 用 boundary check（与 evaluate.py 标准一致）检测是否泄露
          3. 泄露 → 该候选直接作为 rejected（天然是泄露版本），重新生成 chosen
          4. chosen 通过后：
             - 若之前已有泄露候选 → 用第一个泄露候选作 rejected（不再另外生成）
             - 若一次就通过（无泄露候选）→ 另外生成 rejected
          5. 所有重试均泄露 → 返回 None，调用方丢弃此样本
        """
        leaked_candidate: str | None = None  # 保存第一个泄露候选，直接用作 rejected

        for attempt in range(max_retries):
            chosen_candidate = self._generate_preference_chosen(
                name, chapter, state, few_shot, question
            )
            leaked = self._check_boundary_leaked(
                name, chapter, question, chosen_candidate, unknown_fact
            )

            if not leaked:
                if attempt > 0:
                    print(f"   🔄 chosen 第{attempt+1}次重试通过边界检测")
                # 优先用之前泄露的候选作 rejected（语义更自然），否则另外生成
                if leaked_candidate is not None:
                    rejected = leaked_candidate
                else:
                    rejected = self._generate_preference_rejected(
                        name, chapter, state, few_shot, question, unknown_fact
                    )
                return PreferenceItem(
                    character=name,
                    chapter=chapter,
                    unknown_fact=unknown_fact,
                    question=question,
                    chosen=chosen_candidate,
                    rejected=rejected,
                    few_shot=few_shot,
                )
            else:
                # chosen 泄露 → 保存为 rejected 候选，重新生成
                print(f"   ⚠️ chosen 泄露（第{attempt+1}次），保存为 rejected 候选，重新生成 chosen...")
                if leaked_candidate is None:
                    leaked_candidate = chosen_candidate  # 只保留第一个，风格最自然

        # 所有重试均泄露，丢弃此样本
        return None


# ── 输出格式化 ────────────────────────────────────────────────────────────────

def _build_instruction(state: CharacterState, few_shot: str) -> str:
    base = state.to_prompt()
    extra = []
    if few_shot and "暂无" not in few_shot:
        extra.append(f"### 你的原著台词示例（参考这个风格说话）\n{few_shot}")
    if extra:
        return base + "\n\n" + "\n\n".join(extra)
    return base


def to_sft_format(items: list[StyleSFTItem], manager: CharacterStateManager) -> list[dict]:
    records = []
    for item in items:
        state = manager.get_state_at_chapter(item.character, item.chapter)
        if state is None:
            continue
        records.append({
            "instruction": _build_instruction(state, item.few_shot),
            "input": item.question,
            "output": item.answer,
        })
    return records


def to_preference_format(items: list[PreferenceItem], manager: CharacterStateManager) -> list[dict]:
    records = []
    for item in items:
        state = manager.get_state_at_chapter(item.character, item.chapter)
        if state is None:
            continue
        records.append({
            "instruction": _build_instruction(state, item.few_shot),
            "input": item.question,
            "chosen": item.chosen,
            "rejected": item.rejected,
        })
    return records


def save_jsonl(records: list[dict], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"💾 已保存 {len(records)} 条 → {path}")
