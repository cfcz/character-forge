"""
评测脚本：测试模型在角色扮演上的表现

两个核心指标：
  1. 边界保持率：面对诱导性问题时，模型能否不泄露 unknown_facts
  2. 风格得分 / Pairwise Win Rate：回答是否符合角色说话习惯

用法：
  # 单模型评测
  python scripts/evaluate.py \\
    --model-a /root/models/Qwen/Qwen2.5-1.5B-Instruct \\
    --sft data/train/sft_val.jsonl \\
    --preference data/train/dpo_val.jsonl \\
    --judge-provider deepseek --n 30 \\
    --output results/base_eval.json

  # Pairwise 对比评测（base vs SFT）
  python scripts/evaluate.py \\
    --model-a /root/models/Qwen/Qwen2.5-1.5B-Instruct \\
    --model-b /root/output/qwen_sft_full \\
    --sft data/train/sft_val.jsonl \\
    --preference data/train/dpo_val.jsonl \\
    --judge-provider deepseek --n 30 \\
    --output results/pairwise_eval.json
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

STYLE_SINGLE_PROMPT = """\
你是一个文学风格分析师，评估角色扮演回答的风格还原度。

## 角色原著台词示例（参考标准）
{few_shot}

## 用户问题
{question}

## 模型生成的回答
{answer}

## 评分任务
结合用户问题的场景，对比原著台词风格，给模型回答打分（1-5分）：
- 5分：句式、语气、用词几乎和原著一样，完全像这个角色说的
- 4分：风格接近，有一两处不像
- 3分：能看出是这个角色，但有明显的"AI腔"
- 2分：风格差距较大，像通用对话模型
- 1分：完全不像，旁白分析，或格式错误（编号列表/加粗/总结语）

只输出 JSON：
{{"score": 1-5, "reason": "一句话说明"}}
"""

STYLE_PAIRWISE_PROMPT = """\
你是一个角色扮演质量评判员。以下是同一个问题的两条回答，请分别打分并判断哪条更好。

## 角色原著台词示例（风格参考）
{few_shot}

## 用户问题
{question}

## 回答 A
{answer_a}

## 回答 B
{answer_b}

## 评分任务
结合用户问题的场景，将两条回答放在一起对比打分（1-5分）：
- 5分：句式、语气、用词几乎和原著一样，完全像这个角色说的
- 4分：风格接近，有一两处不像
- 3分：能看出是这个角色，但有明显"AI腔"
- 2分：风格差距较大，像通用对话模型
- 1分：完全不像，旁白分析，或格式错误（编号列表/加粗/总结语）

注意：两条回答放在一起对比，分数要体现出差距，不要给相同分数除非真的质量相当。

只输出 JSON：
{{"score_a": 1-5, "score_b": 1-5, "winner": "A" 或 "B" 或 "tie", "reason": "一句话说明两者的核心差异"}}
"""


# ── 模型推理 ──────────────────────────────────────────────────────────────────

def run_model_inference(model_path: str, instruction: str, question: str) -> str:
    """用 transformers 直接加载模型推理，支持 HuggingFace ID 或本地路径。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("❌ 请先安装: pip install transformers torch")
        sys.exit(1)

    if not hasattr(run_model_inference, "_cache"):
        run_model_inference._cache = {}

    if model_path not in run_model_inference._cache:
        print(f"📦 加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
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


# ── 格式硬规则 ────────────────────────────────────────────────────────────────

def check_hard_rules(answer: str) -> bool:
    """
    硬性规则检查，返回 False 表示格式崩坏或出现严重 AI 腔。
    """
    # 规则 1：AI 身份自曝
    if re.search(r"(作为(一个)?(AI|人工智能|语言模型)|我(只是|是)(一个)?(AI|人工智能|语言模型))", answer, re.IGNORECASE):
        return False

    # 规则 2：开头拒绝套话
    if re.search(r"^(抱歉|对不起|很抱歉)[，。！\s]*(我无法|我不能|作为|这是一个)", answer):
        return False

    # 规则 3：结尾助手套话
    if re.search(r"(需要我(帮忙|做些什么)吗|随时提问|很高兴为(您|你)解答|如果(您|你)有任何问题|我乐意效劳)[。？！\?!\s]*$", answer):
        return False

    # 规则 4：无法提供类免责声明
    if re.search(
        r"(请注意[，。]这只是|我无法(为(您|你))?提供|这超出了我的(能力|知识)范围"
        r"|(无法|不便|不能)(透露|提供|回答|说明)(具体)?(细节|信息|内容|答案|情况)?)",
        answer,
    ):
        return False

    # 规则 5：人称混乱（旁白模式）
    if len(answer) > 30 and "我" not in answer:
        if re.search(r"[他她][^们]{0,4}(说|感到|觉得|认为|想|看|做|走|回答|表示)", answer):
            return False

    # 规则 6：Markdown 结构（编号列表 / 加粗标题）
    if re.search(r"^\s*\d+\.\s+\*\*", answer, re.MULTILINE):
        return False

    # 规则 7：兜底有效性
    if not re.search(r"[一-龥]", answer):
        return False

    return True


# ── 边界评测 ──────────────────────────────────────────────────────────────────

def evaluate_boundary(samples: list[dict], model_path: str, judge: LLMClient, n: int, label: str = "") -> dict:
    """评测边界保持率：模型是否泄露 unknown_facts"""
    selected = random.sample(samples, min(n, len(samples)))
    results = []

    tag = f"[{label}] " if label else ""
    print(f"\n🔍 {tag}边界评测（共 {len(selected)} 条）")

    for i, sample in enumerate(selected):
        instruction = sample.get("instruction", "")
        question = sample.get("input", "")

        name = _extract_field(instruction, "角色名") or "角色"
        chapter = _extract_field(instruction, "当前章节") or "?"
        unknown_facts_text = _extract_section(instruction, "未知信息") or "（无）"

        answer = run_model_inference(model_path, instruction, question)

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

        results.append({
            "question": question,
            "answer": answer,
            "unknown_facts": unknown_facts_text,   # ← 方便对照迭代
            "leaked": leaked,
            "reason": reason,
        })
        status = "❌泄露" if leaked else "✅守住"
        print(f"   [{i+1}/{len(selected)}] {status} | Q: {question[:30]}...")

    valid = [r for r in results if r["leaked"] is not None]
    leaked_count = sum(1 for r in valid if r["leaked"])
    boundary_rate = (len(valid) - leaked_count) / len(valid) if valid else 0

    print(f"\n📊 {tag}边界保持率: {boundary_rate:.1%}  ({len(valid)-leaked_count}/{len(valid)} 未泄露)")
    return {"boundary_rate": boundary_rate, "details": results}


# ── 风格评测（单模型） ────────────────────────────────────────────────────────

def evaluate_style_single(samples: list[dict], model_path: str, judge: LLMClient, n: int, label: str = "") -> dict:
    """单模型风格评测：格式准确率 + 风格得分"""
    selected = random.sample(samples, min(n, len(samples)))
    scores = []
    style_scores = []
    format_pass = 0
    details = []

    tag = f"[{label}] " if label else ""
    print(f"\n🎭 {tag}风格评测（共 {len(selected)} 条）")

    for i, sample in enumerate(selected):
        instruction = sample.get("instruction", "")
        question = sample.get("input", "")
        few_shot = _extract_section(instruction, "原著台词示例") or "（无台词示例）"

        answer = run_model_inference(model_path, instruction, question)

        if not check_hard_rules(answer):
            score = 1
            reason = "规则拦截：包含AI套话或格式严重错误"
            scores.append(score)
            details.append({"question": question, "answer": answer, "format_pass": False, "score": score, "reason": reason})
            print(f"   [{i+1}/{len(selected)}] ❌格式 | {score}/5 | {reason}")
            continue

        format_pass += 1
        judge_prompt = STYLE_SINGLE_PROMPT.format(few_shot=few_shot, question=question, answer=answer)
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
        print(f"   [{i+1}/{len(selected)}] ✅格式 | {score}/5 | {reason}")

    total = len(selected)
    format_accuracy = format_pass / total if total else 0
    avg_style_score = sum(style_scores) / len(style_scores) if style_scores else 0
    combined_score = format_accuracy * (avg_style_score / 5.0)

    print(f"\n📊 {tag}格式准确率: {format_accuracy:.1%}  ({format_pass}/{total})")
    print(f"📊 {tag}风格得分:   {avg_style_score:.2f}/5.0")
    print(f"📊 {tag}综合得分:   {combined_score:.3f}")

    return {
        "format_accuracy": format_accuracy,
        "avg_style_score": avg_style_score,
        "combined_score": combined_score,
        "format_pass": format_pass,
        "total": total,
        "details": details,
    }


# ── 风格评测（Pairwise） ──────────────────────────────────────────────────────

def evaluate_style_pairwise(
    samples: list[dict],
    model_a_path: str,
    model_b_path: str,
    judge: LLMClient,
    n: int,
) -> dict:
    """
    Pairwise 风格对比：两个模型在相同问题上各生成回答，judge 同时打分并选出胜者。
    随机交换 A/B 位置以消除位置偏差。
    """
    selected = random.sample(samples, min(n, len(samples)))

    scores_a, scores_b = [], []
    format_pass_a, format_pass_b = 0, 0
    wins_a, wins_b, ties = 0, 0, 0
    details = []

    print(f"\n🎭 Pairwise 风格评测（共 {len(selected)} 条）")

    for i, sample in enumerate(selected):
        instruction = sample.get("instruction", "")
        question = sample.get("input", "")
        few_shot = _extract_section(instruction, "原著台词示例") or "（无台词示例）"

        ans_a = run_model_inference(model_a_path, instruction, question)
        ans_b = run_model_inference(model_b_path, instruction, question)

        fmt_a = check_hard_rules(ans_a)
        fmt_b = check_hard_rules(ans_b)
        if fmt_a:
            format_pass_a += 1
        if fmt_b:
            format_pass_b += 1

        # 随机交换位置，消除 judge 的位置偏差
        swap = random.random() < 0.5
        left_ans, right_ans = (ans_b, ans_a) if swap else (ans_a, ans_b)

        judge_prompt = STYLE_PAIRWISE_PROMPT.format(
            few_shot=few_shot,
            question=question,
            answer_a=left_ans,
            answer_b=right_ans,
        )
        try:
            result = judge.generate_json(judge_prompt, temperature=0.0)
            raw_score_left = int(result.get("score_a", 3))
            raw_score_right = int(result.get("score_b", 3))
            raw_winner = result.get("winner", "tie")
            reason = result.get("reason", "")

            # 还原回真实的 A/B
            if swap:
                score_a, score_b = raw_score_right, raw_score_left
                winner = {"A": "B", "B": "A", "tie": "tie"}.get(raw_winner, "tie")
            else:
                score_a, score_b = raw_score_left, raw_score_right
                winner = raw_winner
        except Exception:
            score_a, score_b, winner, reason = 3, 3, "tie", "judge error"

        scores_a.append(score_a)
        scores_b.append(score_b)
        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        else:
            ties += 1

        details.append({
            "question": question,
            "answer_a": ans_a,
            "answer_b": ans_b,
            "format_pass_a": fmt_a,
            "format_pass_b": fmt_b,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
            "reason": reason,
        })

        win_tag = {"A": "A胜", "B": "B胜", "tie": "平局"}.get(winner, "?")
        print(f"   [{i+1}/{len(selected)}] {win_tag} | A:{score_a}/5  B:{score_b}/5 | {reason[:50]}")

    total = len(selected)
    avg_a = sum(scores_a) / total if total else 0
    avg_b = sum(scores_b) / total if total else 0
    win_rate_b = wins_b / total if total else 0

    print(f"\n📊 model_a 平均分: {avg_a:.2f}/5.0  格式通过: {format_pass_a}/{total}")
    print(f"📊 model_b 平均分: {avg_b:.2f}/5.0  格式通过: {format_pass_b}/{total}")
    print(f"📊 model_b Win Rate: {win_rate_b:.1%}  (A胜{wins_a} / B胜{wins_b} / 平{ties})")

    return {
        "model_a_avg_score": avg_a,
        "model_b_avg_score": avg_b,
        "model_a_format_pass": format_pass_a,
        "model_b_format_pass": format_pass_b,
        "model_b_win_rate": win_rate_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total": total,
        "details": details,
    }


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _extract_field(text: str, field: str) -> str:
    m = re.search(rf"{field}[：:]\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _extract_section(text: str, section: str) -> str:
    m = re.search(rf"###?\s*[^\n]*{re.escape(section)}[^\n]*\n([\s\S]+?)(?=\n###?|\Z)", text)
    return m.group(1).strip() if m else ""


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Character Forge 评测脚本")
    parser.add_argument("--model-a", required=True, help="模型A路径（base 模型 或 单模型评测时的模型）")
    parser.add_argument("--model-b", default=None, help="模型B路径（SFT/DPO 模型，提供时启用 pairwise 模式）")
    parser.add_argument("--preference", type=str, help="Preference JSONL 路径（用于边界评测）")
    parser.add_argument("--sft", type=str, help="SFT JSONL 路径（用于风格评测）")
    parser.add_argument("--judge-provider", default="deepseek", choices=["deepseek", "qwen", "openai"])
    parser.add_argument("--n", type=int, default=30, help="每项评测抽取样本数（默认30）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="保存评测结果的 JSON 路径")
    args = parser.parse_args()

    random.seed(args.seed)
    judge = LLMClient(provider=args.judge_provider)
    pairwise = args.model_b is not None

    all_results = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "mode": "pairwise" if pairwise else "single",
    }

    # ── 边界评测 ──────────────────────────────────────────────────────────────
    if args.preference:
        pref_samples = [json.loads(l) for l in Path(args.preference).read_text().splitlines() if l.strip()]
        all_results["boundary_a"] = evaluate_boundary(pref_samples, args.model_a, judge, args.n, label="A")
        if pairwise:
            random.seed(args.seed)   # 重置 seed，保证两次抽到同一批样本
            all_results["boundary_b"] = evaluate_boundary(pref_samples, args.model_b, judge, args.n, label="B")

    # ── 风格评测 ──────────────────────────────────────────────────────────────
    if args.sft:
        sft_samples = [json.loads(l) for l in Path(args.sft).read_text().splitlines() if l.strip()]
        if pairwise:
            random.seed(args.seed)
            all_results["style_pairwise"] = evaluate_style_pairwise(
                sft_samples, args.model_a, args.model_b, judge, args.n
            )
        else:
            all_results["style"] = evaluate_style_single(sft_samples, args.model_a, judge, args.n)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📋 评测汇总")
    print("=" * 55)

    if "boundary_a" in all_results:
        print(f"  边界保持率 A: {all_results['boundary_a']['boundary_rate']:.1%}")
    if "boundary_b" in all_results:
        print(f"  边界保持率 B: {all_results['boundary_b']['boundary_rate']:.1%}")

    if "style_pairwise" in all_results:
        sp = all_results["style_pairwise"]
        print(f"  风格得分  A: {sp['model_a_avg_score']:.2f}/5.0  格式通过: {sp['model_a_format_pass']}/{sp['total']}")
        print(f"  风格得分  B: {sp['model_b_avg_score']:.2f}/5.0  格式通过: {sp['model_b_format_pass']}/{sp['total']}")
        print(f"  B Win Rate:  {sp['model_b_win_rate']:.1%}  (A胜{sp['wins_a']} / B胜{sp['wins_b']} / 平{sp['ties']})")
    elif "style" in all_results:
        s = all_results["style"]
        print(f"  格式准确率: {s['format_accuracy']:.1%}")
        print(f"  风格得分:   {s['avg_style_score']:.2f}/5.0")
        print(f"  综合得分:   {s['combined_score']:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print(f"\n💾 结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
