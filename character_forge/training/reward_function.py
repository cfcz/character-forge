"""
Character Forge — GRPO Reward Function
=======================================
三层 reward 设计：
  0.0  格式崩坏（AI 腔、旁白、拒绝套话等）
  0.2  边界泄露（response entails unknown_fact）
  0.5  幻觉（response contradicts known_facts）
  1.0  全部通过

NLI 模型：IDEA-CCNL/Erlangshen-Roberta-110M-NLI
  - 110M 参数，中文 NLI，4090 上单次推理 <10ms
  - 只在第一次调用时加载，之后复用

LLaMA-Factory GRPO 接口：
  reward_funcs 参数接收一个函数列表，每个函数签名为：
    fn(completions: list[str], **kwargs) -> list[float]
  kwargs 里包含 dataset 的其他字段（instruction、input 等）
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── NLI 模型（全局单例，只加载一次）────────────────────────────────────────────

NLI_MODEL_NAME = "/root/models/Erlangshen-Roberta-110M-NLI"
NLI_ENTAIL_THRESHOLD = 0.7      # 超过此置信度才判定为"蕴含"
NLI_CONTRADICT_THRESHOLD = 0.7  # 超过此置信度才判定为"矛盾"

_nli_tokenizer = None
_nli_model = None
_label2idx: dict[str, int] = {}   # {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_nli_model():
    global _nli_tokenizer, _nli_model, _label2idx
    if _nli_model is not None:
        return

    print(f"[RewardFn] 加载 NLI 模型: {NLI_MODEL_NAME} ...")
    _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    _nli_model.eval()
    if torch.cuda.is_available():
        _nli_model = _nli_model.cuda()

    # 从 config 中解析 label → index 映射，兼容不同模型的 label 顺序
    id2label = _nli_model.config.id2label
    for idx, label in id2label.items():
        label_lower = label.lower()
        if "entail" in label_lower:
            _label2idx["entailment"] = int(idx)
        elif "contradict" in label_lower:
            _label2idx["contradiction"] = int(idx)
        elif "neutral" in label_lower:
            _label2idx["neutral"] = int(idx)

    print(f"[RewardFn] NLI 模型加载完成，label 映射: {_label2idx}")


def _nli_predict(premise: str, hypothesis: str) -> dict[str, float]:
    """
    对一对 (premise, hypothesis) 运行 NLI，返回各类别概率。
    截断到 512 token，避免超长文本报错。
    """
    _load_nli_model()

    inputs = _nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = _nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu()

    return {
        "entailment":    probs[_label2idx.get("entailment", 0)].item(),
        "neutral":       probs[_label2idx.get("neutral", 1)].item(),
        "contradiction": probs[_label2idx.get("contradiction", 2)].item(),
    }


# ── 格式硬规则（与 evaluate.py 保持一致）───────────────────────────────────────

def check_hard_rules(answer: str) -> bool:
    """返回 False 表示格式崩坏或出现严重 AI 腔。"""
    # AI 身份自曝
    if re.search(
        r"(作为(一个)?(AI|人工智能|语言模型)|我(只是|是)(一个)?(AI|人工智能|语言模型))",
        answer, re.IGNORECASE
    ):
        return False
    # 开头拒绝套话
    if re.search(r"^(抱歉|对不起|很抱歉)[，。！\s]*(我无法|我不能|作为|这是一个)", answer):
        return False
    # 结尾助手套话
    if re.search(
        r"(需要我(帮忙|做些什么)吗|随时提问|很高兴为(您|你)解答"
        r"|如果(您|你)有任何问题|我乐意效劳)[。？！\?!\s]*$",
        answer,
    ):
        return False
    # 无法提供类免责声明
    if re.search(
        r"(请注意[，。]这只是|我无法(为(您|你))?提供|这超出了我的(能力|知识)范围"
        r"|(无法|不便|不能)(透露|提供|回答|说明)(具体)?(细节|信息|内容|答案|情况)?)",
        answer,
    ):
        return False
    # 第三人称旁白（人称混乱）
    if len(answer) > 30 and "我" not in answer:
        if re.search(r"[他她][^们]{0,4}(说|感到|觉得|认为|想|看|做|走|回答|表示)", answer):
            return False
    # Markdown 编号列表加粗
    if re.search(r"^\s*\d+\.\s+\*\*", answer, re.MULTILINE):
        return False
    return True


# ── Instruction 解析 ───────────────────────────────────────────────────────────

def _extract_section(instruction: str, section: str) -> str:
    """从 instruction 中提取指定 ### 小节的内容（与 evaluate.py 保持一致）。"""
    pattern = rf"###?\s*[^\n]*{re.escape(section)}[^\n]*\n(.*?)(?=\n###|\Z)"
    m = re.search(pattern, instruction, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_bullet_list(text: str) -> list[str]:
    """把 markdown bullet list 解析为字符串列表。"""
    items = []
    for line in text.splitlines():
        line = re.sub(r"^[-*•]\s*", "", line).strip()
        if line:
            items.append(line)
    return items


# ── 核心 reward 计算 ───────────────────────────────────────────────────────────

def compute_reward(instruction: str, response: str) -> float:
    """
    计算单条 response 的 reward。

    Args:
        instruction: 完整的角色状态 instruction（含已知/未知信息）
        response:    模型生成的回答

    Returns:
        0.0  格式不通过
        0.2  边界泄露（response entails 某条 unknown_fact）
        0.5  幻觉（response contradicts known_facts）
        1.0  全部通过
    """
    response = response.strip()
    if not response:
        return 0.0

    # ── 1. 格式检查 ────────────────────────────────────────────────────────────
    if not check_hard_rules(response):
        return 0.0

    # ── 解析角色状态 ───────────────────────────────────────────────────────────
    unknown_text = _extract_section(instruction, "未知信息")
    known_text   = _extract_section(instruction, "已知信息")
    unknown_facts = _parse_bullet_list(unknown_text)

    # ── 2. 边界泄露检查 ────────────────────────────────────────────────────────
    # 逐条 unknown_fact 检测：response 是否蕴含了它？
    for fact in unknown_facts:
        if not fact or len(fact) < 4:
            continue
        result = _nli_predict(premise=response, hypothesis=fact)
        if result["entailment"] > NLI_ENTAIL_THRESHOLD:
            return 0.2

    # ── 3. 幻觉检测 ────────────────────────────────────────────────────────────
    # 检测 response 是否与 known_facts 明确矛盾（编造了相反的事实）
    if known_text:
        result = _nli_predict(premise=known_text, hypothesis=response)
        if result["contradiction"] > NLI_CONTRADICT_THRESHOLD:
            return 0.5

    # ── 全部通过 ───────────────────────────────────────────────────────────────
    return 1.0


# ── LLaMA-Factory GRPO 接口 ────────────────────────────────────────────────────

def character_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    LLaMA-Factory GRPO reward_funcs 接口。

    LLaMA-Factory 在调用时会把 dataset 的其他列通过 kwargs 传入。
    我们需要 'instruction' 列，格式与 SFT/DPO 数据一致。

    用法（在训练脚本或 yaml 里指定）：
        reward_funcs:
          - character_forge.training.reward_function.character_reward_fn
    """
    instructions = kwargs.get("instruction", [""] * len(completions))

    rewards = []
    for instruction, response in zip(instructions, completions):
        try:
            r = compute_reward(instruction, response)
        except Exception as e:
            print(f"[RewardFn] ⚠️ reward 计算失败，默认 0.0: {e}")
            r = 0.0
        rewards.append(r)

    return rewards
