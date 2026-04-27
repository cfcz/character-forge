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
2. 可以提到其他角色，但只能问"{name}"怎么看、怎么想、怎么经历，不能把其他角色当成被提问者
3. 问题要像正常聊天，不要写成角色回答口吻，不要写成旁白
4. 不要出现"..."、"……"、括号动作
5. 不要用固定句式重复提问，尽量自然

只输出 JSON 数组，例如：["问题1", "问题2"]
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
- 结尾加感悟总结
- 原样复制 few-shot 里的句子
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
- 句长、语气、用词贴近上方台词示例
- 不要括号动作、神态描写、心理描写、舞台说明
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


def _extract_dialogues_by_speaker(name: str, text: str, max_lines: int) -> list[str]:
    """
    严格抽取：只保留有明确"{name}+说话动词"证据的引号内容。
    """
    lines: list[str] = []
    for m in re.finditer(r"“([^”]{{2,160}})”", text):
        quote = _clean_dialogue_line(m.group(1).strip())
        if len(quote) < 3:
            continue
        left = text[max(0, m.start() - 90):m.start()]
        right = text[m.end():min(len(text), m.end() + 90)]

        if _has_target_speaker_evidence(name, left, right):
            lines.append(quote)
            continue

        # 明确检测到"其他人说"，并且没有目标说话证据，直接丢弃
        if _has_other_speaker_evidence(name, left, right):
            continue

    return _dedupe_keep_order(lines)[:max_lines]


def _line_has_target_speaker_support(name: str, line: str, text: str) -> bool:
    """
    校验某条台词是否能在原文引号中找到，且具备目标说话人证据。
    """
    for m in re.finditer(r"“([^”]{{2,200}})”", text):
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
    for m in re.finditer(r"“([^”]{{2,120}})”", text):
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
    ) -> list[StyleSFTItem]:
        chapters = split_chapters(novel_text)
        chapter_map = {ch.index: ch for ch in chapters}
        total_chapters = self._get_total_chapters(manager, max_chapters)
        target_chars = characters or manager.list_characters()

        all_items: list[StyleSFTItem] = []
        for char_name in target_chars:
            if max_samples and len(all_items) >= max_samples:
                break
            print(f"\n🎭 [SFT] 处理角色：{char_name}")
            remaining = (max_samples - len(all_items)) if max_samples else None
            items = self._synthesize_sft_character(char_name, manager, chapter_map, total_chapters, max_samples=remaining)
            all_items.extend(items)
            print(f"   ✅ 生成 {len(items)} 条 SFT 样本")
        return all_items[:max_samples] if max_samples else all_items

    def synthesize_preference(
        self,
        manager: CharacterStateManager,
        novel_text: str,
        characters: list[str] | None = None,
        max_chapters: int | None = None,
        max_samples: int | None = None,   # dry-run 用：凑够就停
    ) -> list[PreferenceItem]:
        chapters = split_chapters(novel_text)
        chapter_map = {ch.index: ch for ch in chapters}
        total_chapters = self._get_total_chapters(manager, max_chapters)
        target_chars = characters or manager.list_characters()

        all_items: list[PreferenceItem] = []
        for char_name in target_chars:
            if max_samples and len(all_items) >= max_samples:
                break
            print(f"\n🎭 [Preference] 处理角色：{char_name}")
            remaining = (max_samples - len(all_items)) if max_samples else None
            items = self._synthesize_preference_character(char_name, manager, chapter_map, total_chapters, max_samples=remaining)
            all_items.extend(items)
            print(f"   ✅ 生成 {len(items)} 条 Preference 样本")
        return all_items[:max_samples] if max_samples else all_items

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
    ) -> list[StyleSFTItem]:
        items: list[StyleSFTItem] = []

        last_state_chapter = -1  # 记录上一个切片的 state.chapter，避免重复处理
        for chapter_idx in self._build_checkpoints(total_chapters):
            if max_samples and len(items) >= max_samples:
                break
            state = manager.get_state_at_chapter(name, chapter_idx)
            if state is None:
                print(f"   第{chapter_idx}章：角色不存在，跳过")
                continue
            if not self._has_sft_material(state):
                print(f"   第{chapter_idx}章：缺少 SFT 素材，跳过")
                continue
            # 跳过状态没有变化的切片（角色在这段时间没出现在 character_changes 里）
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
                if max_samples and len(items) >= max_samples:
                    break
                try:
                    answer = self._generate_sft_answer(name, chapter_idx, state, few_shot, question)
                    items.append(StyleSFTItem(
                        character=name,
                        chapter=chapter_idx,
                        question=question,
                        answer=answer,
                        few_shot=few_shot,
                    ))
                except Exception as e:
                    print(f"   ⚠️ SFT 样本生成失败（{name} 第{chapter_idx}章）: {e}")
        return items

    def _synthesize_preference_character(
        self,
        name: str,
        manager: CharacterStateManager,
        chapter_map: dict[int, Chapter],
        total_chapters: int,
        max_samples: int | None = None,
    ) -> list[PreferenceItem]:
        items: list[PreferenceItem] = []

        last_state_chapter = -1
        for chapter_idx in self._build_checkpoints(total_chapters):
            if max_samples and len(items) >= max_samples:
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
                if max_samples and len(items) >= max_samples:
                    break
                try:
                    questions = self._generate_preference_questions(name, chapter_idx, fact)
                    for question in questions:
                        if max_samples and len(items) >= max_samples:
                            break
                        chosen = self._generate_preference_chosen(
                            name, chapter_idx, state, few_shot, question
                        )
                        rejected = self._generate_preference_rejected(
                            name, chapter_idx, state, few_shot, question, fact
                        )
                        items.append(PreferenceItem(
                            character=name,
                            chapter=chapter_idx,
                            unknown_fact=fact,
                            question=question,
                            chosen=chosen,
                            rejected=rejected,
                            few_shot=few_shot,
                        ))
                except Exception as e:
                    print(f"   ⚠️ Preference 样本生成失败（{name} 第{chapter_idx}章）: {e}")
        return items

    def _has_sft_material(self, state: CharacterState) -> bool:
        # 至少要有一定量的已知信息或关键记忆，否则生成的回答天花板太低
        has_enough_facts = len(state.known_facts) >= 2 or len(state.key_memories) >= 1
        has_any = any([state.goals, state.relationships, state.emotional_state])
        return has_enough_facts and has_any

    def _build_sft_intent_plan(self, chapter: int) -> list[tuple[str, str, int]]:
        intents: list[tuple[str, str]] = [
            ("事实追问", "围绕已知事实或关键记忆，请角色补充细节、核对认知"),
            ("情绪追问", "围绕当前情绪或压力体验，问角色当下心理状态"),
            ("关系追问", "围绕与其他人物的关系变化，问角色态度和看法"),
            ("观点追问", "围绕事件判断或价值倾向，问角色如何理解当前局势"),
            ("行动追问", "围绕当前目标与下一步行动，问角色接下来打算"),
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
        return False

    def _is_bad_generated_answer(self, text: str) -> bool:
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
        return False

    def _generate_answer_with_retry(self, prompt: str, base_temperature: float) -> str:
        last = ""
        for i in range(self.regen_retries + 1):
            temp = min(base_temperature + 0.08 * i, 0.75)
            raw = self.llm.generate(prompt, temperature=temp).strip()
            cleaned = _clean_generated_answer(raw)
            last = cleaned
            if not self._is_bad_generated_answer(cleaned):
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

        # 先走严格 speaker 规则，避免把"别人对{name}说的话"抽进 few-shot
        strict_lines = _extract_dialogues_by_speaker(name, combined, self.dialogue_examples)
        if len(strict_lines) >= min(4, self.dialogue_examples):
            result = "\n".join(strict_lines[:self.dialogue_examples])
            self._dialogue_cache[cache_key] = result
            return result

        prompt = DIALOGUE_EXTRACT_PROMPT.format(
            name=name,
            chapter_content=combined,
            n_examples=self.dialogue_examples,
        )

        try:
            raw = self.llm.generate(prompt, temperature=0.1).strip()
            if raw == "无" or not raw:
                llm_lines: list[str] = []
            else:
                cleaned = _clean_dialogues(raw, max_lines=self.dialogue_examples * 2)
                llm_lines = [line.strip() for line in cleaned.splitlines() if line.strip()] if cleaned else []

            # 用 LLM 验证步骤替代纯正则校验，处理代词归属等复杂情况
            validated_llm = self._validate_dialogues_with_llm(name, llm_lines, combined)

            merged = _dedupe_keep_order(strict_lines + validated_llm)[:self.dialogue_examples]
            if merged:
                result = "\n".join(merged)
            else:
                # 不再退回"名字邻近"启发式，避免错把别人台词归到当前角色
                result = fallback
        except Exception as e:
            print(f"   ⚠️ 台词提取失败: {e}")
            if strict_lines:
                result = "\n".join(strict_lines[:self.dialogue_examples])
            else:
                result = fallback

        self._dialogue_cache[cache_key] = result
        return result

    def _validate_dialogues_with_llm(
        self,
        name: str,
        candidates: list[str],
        chapter_content: str,
    ) -> list[str]:
        """
        使用 LLM 验证候选台词是否真的属于目标角色，处理代词归属等复杂情况。
        只有当候选列表非空时才调用 LLM。
        """
        if not candidates:
            return []

        # 先用正则快速过滤掉明确不属于目标角色的（速度快，避免不必要的 LLM 调用）
        # 只保留"无明确其他说话人证据"的，再交给 LLM 做最终判断
        candidates_no_other = []
        for line in candidates:
            # 在原文里找到这行台词的位置，检查是否有其他说话人证据
            found_other = False
            for m in re.finditer(r"“([^”]{{2,200}})”", chapter_content):
                quote = _clean_dialogue_line(m.group(1).strip())
                if not quote:
                    continue
                if line not in quote and quote not in line:
                    continue
                left = chapter_content[max(0, m.start() - 90):m.start()]
                right = chapter_content[m.end():min(len(chapter_content), m.end() + 90)]
                if _has_other_speaker_evidence(name, left, right):
                    found_other = True
                    break
            if not found_other:
                candidates_no_other.append(line)

        if not candidates_no_other:
            return []

        # 截取章节内容（避免 prompt 过长）
        content_for_validate = chapter_content[:6000]

        validate_prompt = DIALOGUE_VALIDATE_PROMPT.format(
            name=name,
            dialogue_list=json.dumps(candidates_no_other, ensure_ascii=False),
            chapter_content=content_for_validate,
        )

        try:
            result = self.llm.generate_json(validate_prompt, temperature=0.0)
            if isinstance(result, list):
                validated = [str(line).strip() for line in result if isinstance(line, str) and str(line).strip()]
                return validated
        except Exception as e:
            print(f"   ⚠️ 台词验证失败，保留未过滤候选: {e}")
            return candidates_no_other

        return candidates_no_other

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
        return self._generate_answer_with_retry(prompt, base_temperature=0.35)

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
        return self._generate_answer_with_retry(prompt, base_temperature=0.35)

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
        return self._generate_answer_with_retry(prompt, base_temperature=0.6)


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
