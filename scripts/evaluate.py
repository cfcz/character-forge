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


def evaluate_style(samples: list[dict], model_path: str, judge: LLMClient, n: int) -> dict:
    """评测风格相似度"""
    selected = random.sample(samples, min(n, len(samples)))
    scores = []

    print(f"\n🎭 风格评测（共 {len(selected)} 条）")
    for i, sample in enumerate(selected):
        instruction = sample.get("instruction", "")
        question = sample.get("input", "")
        few_shot = _extract_section(instruction, "原著台词示例") or "（无台词示例）"

        answer = run_model_inference(model_path, instruction, question)

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
        print(f"   [{i+1}/{len(selected)}] 风格得分: {score}/5 | {reason}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n📊 平均风格得分: {avg_score:.2f}/5.0")
    return {"avg_style_score": avg_score, "scores": scores}


def _extract_field(text: str, field: str) -> str:
    """从 instruction 文本中提取简单字段"""
    import re
    m = re.search(rf"{field}[：:]\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _extract_section(text: str, section: str) -> str:
    """从 instruction 文本中提取段落"""
    import re
    m = re.search(rf"###?\s*{section}\s*\n([\s\S]+?)(?=\n###?|\Z)", text)
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
        print(f"  风格得分:   {all_results['style']['avg_style_score']:.2f} / 5.0")

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print(f"\n💾 结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
