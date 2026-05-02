"""
用 reward_function 的 NLI 模型重新评估 evaluation JSON 中两个模型的回答。
对比 NLI 判定结果 vs LLM judge 结果，找出分歧点。

用法：
  python scripts/score_with_reward.py results/sft_vs_grpo.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from character_forge.training.reward_function import (
    _nli_predict,
    NLI_ENTAIL_THRESHOLD,
    check_hard_rules,
)


def score_answer(answer: str, unknown_facts_text: str) -> dict:
    """用 NLI 对单条回答做边界检查，返回 reward 和原因。"""
    answer = answer.strip()
    if not answer:
        return {"reward": 0.0, "reason": "空回答"}

    if not check_hard_rules(answer):
        return {"reward": 0.0, "reason": "格式不通过"}

    facts = []
    for line in unknown_facts_text.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if line and len(line) >= 4:
            facts.append(line)

    leaked_facts = []
    for fact in facts:
        result = _nli_predict(premise=answer, hypothesis=fact)
        if result["entailment"] > NLI_ENTAIL_THRESHOLD:
            leaked_facts.append((fact, result["entailment"]))

    if leaked_facts:
        fact_str = leaked_facts[0][0][:40]
        score = leaked_facts[0][1]
        return {"reward": 0.2, "reason": f"泄漏: {fact_str}... (p={score:.3f})"}

    return {"reward": 1.0, "reason": "通过"}


def analyze(details: list[dict], label: str):
    print(f"\n{'='*65}")
    print(f"[{label}]  NLI Reward 分析  ({len(details)} 条)")
    print(f"{'='*65}")

    scores = []
    disagreements = 0

    for i, item in enumerate(details):
        answer       = item.get("answer", "")
        unknown_facts = item.get("unknown_facts", "")
        question     = item.get("question", "")
        llm_leaked   = item.get("leaked", None)

        nli = score_answer(answer, unknown_facts)
        scores.append(nli["reward"])
        nli_leaked = nli["reward"] == 0.2

        agree_tag = ""
        if llm_leaked is not None and nli_leaked != llm_leaked:
            agree_tag = "  ⚠️ 与LLM判定不一致"
            disagreements += 1

        print(f"\n[{i+1}]  Q: {question[:55]}...")
        print(f"  A: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print(f"  NLI  → reward={nli['reward']}  {nli['reason']}")
        llm_str = "泄漏" if llm_leaked else ("未泄漏" if llm_leaked is False else "N/A")
        print(f"  LLM  → {llm_str}{agree_tag}")

    total = len(scores)
    leaked_n = sum(1 for s in scores if s == 0.2)
    avg = sum(scores) / total if total else 0

    print(f"\n{'─'*65}")
    print(f"📊 [{label}] NLI 平均 reward  : {avg:.3f}")
    print(f"📊 [{label}] NLI 判定泄漏     : {leaked_n}/{total}")
    print(f"📊 [{label}] NLI 边界保持率   : {(total - leaked_n) / total:.1%}")
    print(f"📊 [{label}] 与 LLM judge 分歧: {disagreements}/{total}")


def main():
    eval_path = sys.argv[1] if len(sys.argv) > 1 else "results/sft_vs_grpo.json"
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    for model_key, label in [("boundary_a", "SFT"), ("boundary_b", "GRPO")]:
        if model_key not in data:
            print(f"⚠️ 找不到 {model_key}，跳过")
            continue
        analyze(data[model_key]["details"], label)

    # ── 汇总对比 ──────────────────────────────────────────────────────────────
    if "boundary_a" in data and "boundary_b" in data:
        def boundary_rate(details):
            scores = [score_answer(d["answer"], d["unknown_facts"])["reward"] for d in details]
            leaked = sum(1 for s in scores if s == 0.2)
            return (len(scores) - leaked) / len(scores) if scores else 0

        rate_a = boundary_rate(data["boundary_a"]["details"])
        rate_b = boundary_rate(data["boundary_b"]["details"])
        print(f"\n{'='*65}")
        print(f"📋 NLI 边界保持率对比")
        print(f"   SFT  : {rate_a:.1%}")
        print(f"   GRPO : {rate_b:.1%}")
        diff = rate_b - rate_a
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"   差值 : {arrow} {abs(diff):.1%}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
