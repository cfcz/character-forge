"""
评测脚本：测试模型在角色扮演上的表现

两个核心指标：
  1. 边界保持率 (boundary_score)：面对诱导性问题时，模型能否不泄露 unknown_facts
  2. 风格相似度 (style_score)：回答是否符合角色说话习惯

用法：
  # 评测本地模型（SFT 前后对比）
  python scripts/evaluate.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --preference data/synthesis/three_body_preference.jsonl \
    --sft data/synthesis/three_body_sft.jsonl \
    --judge-provider deepseek \
    --n 30

  # 评测 SFT 后的模型（指定本地路径）
  python scripts/evaluate.py \
    --model ./output/qwen_sft \
    --preference data/synthesis/three_body_preference.jsonl \
    --sft data/synthesis/three_body_sft.jsonl \
    --judge-provider deepseek \
    --n 30
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import random
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from character_forge.utils.llm import LLMClient

# ── Judge Prompts ─────────────────────────────────────────────────────────────

BOUNDARY_JUDGE_PROMPT = """\
你是一个严格的裁判，负责判断角色扮演 AI 是否泄露了"不该知道的信息"。

## 角色设定
角色名：{name}
当前时间点：第{chapter}章
该角色此时尚不知情的信息（未知信息）：
{unknown_facts}

## 用户提问
{question}

## 模型回答
{answer}

## 判断任务
请判断：模型的回答是否涉及或暗示了上面列出的"未知信息"？

判断标准：
- 如果回答明确提到了未知信息的内容 → 泄露
- 如果回答用隐晦方式暗示了未知信息 → 泄露
- 如果回答合理回避，或只说"不知道/不确定" → 未泄露
- 如果问题与未知信息无关，回答正常 → 未泄露

只输出 JSON：
{{"leaked": true/false, "reason": "一句话说明原因"}}
"""

STYLE_JUDGE_PROMPT = """\
你是一个文学风格分析师，评估角色扮演回答的风格还原度。

## 角色原著台词示例（参考标准）
{few_shot}

## 模型生成的回答
{answer}

## 评分任务
对比原著台词风格，给模型回答打分（1-5分）：
- 5分：句式、语气、用词几乎和原著一样，完全像这个角色说的
- 4分：风格接近，有一两处不像
- 3分：能看出是这个角色，但有明显的"AI腔"
- 2分：风格差距较大，像通用对话模型
- 1分：完全不像，或有括号动作/旁白

只输出 JSON：
{{"score": 1-5, "reason": "一句话说明"}}
"""


# ── 模型推理 ──────────────────────────────────────────────────────────────────

def run_model_inference(model_path: str, instruction: str, question: str) -> str:
    """
    用 transformers 直接加载模型推理。
    支持 HuggingFace model ID 或本地路径。
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("❌ 请先安装: pip install transformers torch")
        sys.exit(1)

    # 缓存模型，避免重复加载
    if not hasattr(run_model_inference, "_cache"):
        run_model_inference._cache = {}

    if model_path not in run_model_inference._cache:
        print(f"📦 加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        run_model_inference._cache[model_path] = (tokenizer, model)

    tokenizer, model = run_model_inference._cache[model_path]

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── 评测逻辑 ──────────────────────────────────────────────────────────────────

def evaluate_boundary(samples: list[dict], model_path: str, judge: LLMClient, n: int) -> dict:
    """评测边界保持率：模型是否泄露 unknown_facts"""
    selected = random.sample(samples, min(n, len(samples)))
    results = []

    print(f"\n🔍 边界评测（共 {len(selected)} 条）")
    for i, sample in enumerate(selected):
        instruction = sample.get("instruction", "")
        question = sample.get("input", "")

        # 从 instruction 里解析角色名和 unknown_facts
        # instruction 格式参考 _build_instruction
        name = _extract_field(instruction, "角色名") or "角色"
        chapter = _extract_field(instruction, "当前章节") or "?"
        unknown_facts_text = _extract_section(instruction, "未知信息") or "（无）"

        # 用被测模型生成回答
        answer = run_model_inference(model_path, instruction, question)

        # 用 judge 模型判断是否泄露
        judge_prompt = BOUNDARY_JUDGE_PROMPT.format(
            name=name,
            chapter=chapter,
            unknown_facts=unknown_facts_text,
            question=question,
            answer=answer,
        )
        try:
            result = judge.generate_json(judge_prompt, temperature=0.0)
            leaked = result.get("leaked", False)
            reason = result.get("reason", "")
        except Exception as e:
            print(f"   ⚠️ judge 失败: {e}")
            leaked = None
            reason = "judge error"

        results.append({"leaked": leaked, "reason": reason, "answer": answer, "question": question})
        status = "❌泄露" if leaked else "✅守住"
        print(f"   [{i+1}/{len(selected)}] {status} | Q: {question[:30]}...")

    valid = [r for r in results if r["leaked"] is not None]
    leaked_count = sum(1 for r in valid if r["leaked"])
    boundary_rate = (len(valid) - leaked_count) / len(valid) if valid else 0

    print(f"\n📊 边界保持率: {boundary_rate:.1%}  ({len(valid)-leaked_count}/{len(valid)} 未泄露)")
    return {"boundary_rate": boundary_rate, "details": results}

def check_hard_rules(answer: str, is_ancient_setting: bool = False) -> bool:
    """
    硬性规则检查：使用正则短语和位置锚点，精准拦截 AI 套话，降低误伤率。
    返回 False 表示触发硬规则（格式崩坏或出现严重 AI 腔）。
    """
    # 规则 1：绝对的 AI 身份自曝
    ai_identity_pattern = r"(作为(一个)?(AI|人工智能|语言模型)|我(只是|是)(一个)?(AI|人工智能|语言模型))"
    if re.search(ai_identity_pattern, answer, re.IGNORECASE):
        return False

    # 规则 2：典型的开头拒绝套话
    refusal_start_pattern = r"^(抱歉|对不起|很抱歉)[\uff0c。！\s]*(我无法|我不能|作为|这是一个)"
    if re.search(refusal_start_pattern, answer):
        return False

    # 规则 3：典型的结尾助手套话
    assistant_end_pattern = r"(需要我(帮忙|做些什么)吗|随时提问|很高兴为(您|你)解答|如果(您|你)有任何问题|我乐意效劳)[\u3002\uff1f\uff01\?!\s]*$"
    if re.search(assistant_end_pattern, answer):
        return False

    # 规则 4：无法提供类免责声明（含不以"抱歉"开头的句式）
    disclaimer_pattern = (
        r"(请注意[，。]这只是"
        r"|我无法(为(您|你))?提供"
        r"|这超出了我的(能力|知识)范围"
        r"|(无法|不便|不能)(透露|提供|回答|说明)(具体)?(细节|信息|内容|答案|情况)?)"
    )
    if re.search(disclaimer_pattern, answer):
        return False

    # 规则 5：人称混乱 — 角色扮演必须用第一人称，若回答超过 30 字却完全没有"我"，
    # 且存在明显的第三人称叙述（他/她 + 动词），判定为旁白模式
    if len(answer) > 30 and "我" not in answer:
        third_person_pattern = r"[他她][^们]{0,4}(说|感到|觉得|认为|想|看|做|走|回答|表示)"
        if re.search(third_person_pattern, answer):
            return False

    # 规则 6：兜底有效性校验
    if not re.search(r'[一-龥]', answer):
        return False

    return True

def evaluate_style(samples: list[dict], model_path: str, judge: LLMClient, n: int) -> dict:
    """评测格式准确率 + 风格相似度"""
    selected = random.sample(samples, min(n, len(samples)))
    scores = []          # 所有样本的风格分（格式不过关的记 1 分）
    style_scores = []    # 仅格式通过样本的风格分（用于纯风格均值）
    format_pass = 0
    details = []         # 每条样本的完整记录

    print(f"\n🎭 风格评测（共 {len(selected)} 条）")
    for i, sample in enumerate(selected):
        instruction = sample.get("instruction", "")
        question = sample.get("input", "")
        few_shot = _extract_section(instruction, "原著台词示例") or "（无台词示例）"

        answer = run_model_inference(model_path, instruction, question)

        # 1. 先进行正则/规则硬拦截
        if not check_hard_rules(answer):
            score = 1
            reason = "规则拦截：包含AI套话或格式严重错误"
            scores.append(score)
            details.append({"question": question, "answer": answer, "format_pass": False, "score": score, "reason": reason})
            print(f"   [{i+1}/{len(selected)}] ❌格式 | 风格: {score}/5 | {reason}")
            continue  # 直接跳过 LLM 裁判，省钱！

        format_pass += 1

        judge_prompt = STYLE_JUDGE_PROMPT.format(
            few_shot=few_shot,
            answer=answer,
        )
        try:
            result = judge.generate_json(judge_prompt, temperature=0.0)
            score = int(result.get("score", 3))
            reason = result.get("reason", "")
        except Exception:
            score = 3
            reason = "judge error"

        scores.append(score)
        style_scores.append(score)
        details.append({"question": question, "answer": answer, "format_pass": True, "score": score, "reason": reason})
        print(f"   [{i+1}/{len(selected)}] ✅格式 | 风格: {score}/5 | {reason}")

    total = len(selected)
    format_accuracy = format_pass / total if total else 0
    avg_style_score = sum(style_scores) / len(style_scores) if style_scores else 0
    # 综合得分：格式准确率 × 归一化风格分（风格满分5→1.0）
    combined_score = format_accuracy * (avg_style_score / 5.0)

    print(f"\n📊 格式准确率: {format_accuracy:.1%}  ({format_pass}/{total} 通过)")
    print(f"📊 风格得分:   {avg_style_score:.2f}/5.0  （仅格式通过样本）")
    print(f"📊 综合得分:   {combined_score:.3f}  （格式准确率 × 风格归一化）")

    return {
        "format_accuracy": format_accuracy,
        "avg_style_score": avg_style_score,
        "combined_score": combined_score,
        "format_pass": format_pass,
        "total": total,
        "scores": scores,
        "details": details,
    }


def _extract_field(text: str, field: str) -> str:
    """从 instruction 文本中提取简单字段"""
    import re
    m = re.search(rf"{field}[：:]\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _extract_section(text: str, section: str) -> str:
    """从 instruction 文本中提取段落（section 为 header 的子串即可匹配）"""
    import re
    m = re.search(rf"###?\s*[^\n]*{re.escape(section)}[^\n]*\n([\s\S]+?)(?=\n###?|\Z)", text)
    return m.group(1).strip() if m else ""


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Character Forge 评测脚本")
    parser.add_argument("--model", required=True, help="被测模型路径或 HuggingFace ID")
    parser.add_argument("--preference", type=str, help="Preference JSONL 路径（用于边界评测）")
    parser.add_argument("--sft", type=str, help="SFT JSONL 路径（用于风格评测）")
    parser.add_argument("--judge-provider", default="deepseek", choices=["deepseek", "qwen", "openai"])
    parser.add_argument("--n", type=int, default=30, help="每项评测抽取样本数（默认30）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="保存评测结果的 JSON 路径")
    args = parser.parse_args()

    random.seed(args.seed)
    judge = LLMClient(provider=args.judge_provider)

    all_results = {"model": args.model}

    if args.preference:
        pref_samples = [json.loads(l) for l in Path(args.preference).read_text().splitlines() if l.strip()]
        boundary_result = evaluate_boundary(pref_samples, args.model, judge, args.n)
        all_results["boundary"] = boundary_result

    if args.sft:
        sft_samples = [json.loads(l) for l in Path(args.sft).read_text().splitlines() if l.strip()]
        style_result = evaluate_style(sft_samples, args.model, judge, args.n)
        all_results["style"] = style_result

    # 汇总
    print("\n" + "=" * 50)
    print("📋 评测汇总")
    print("=" * 50)
    if "boundary" in all_results:
        print(f"  边界保持率: {all_results['boundary']['boundary_rate']:.1%}")
    if "style" in all_results:
        s = all_results["style"]
        print(f"  格式准确率: {s['format_accuracy']:.1%}  ({s['format_pass']}/{s['total']})")
        print(f"  风格得分:   {s['avg_style_score']:.2f} / 5.0")
        print(f"  综合得分:   {s['combined_score']:.3f}")

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print(f"\n💾 结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
