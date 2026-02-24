
import argparse
import json
import os
import random
from typing import Dict, Any, List, Optional

from src.utils.answer_parser import extract_final_answer_strict

# --- Prompt builders (kept minimal / mirrors your eval prompts) ---

def build_solve_prompt(question: str, dataset: str = "") -> str:
    dataset = (dataset or "").lower()
    if dataset == "math":
        return (
            "Solve the problem. Be concise.\n"
            "Requirements:\n"
            "- Write a short solution (no extra commentary).\n"
            "- Put ONLY the final answer on the last line as \\boxed{...}.\n"
            "- Do not write anything after the final line.\n\n"
            f"Problem:\n{question}\n\n"
            "Solution:\n"
        )
    return (
        "You are a helpful math tutor. Solve the problem step by step.\n"
        "IMPORTANT:\n"
        "- The FINAL line of your output must be exactly: #### <answer>\n"
        "- Do NOT write '####' anywhere except the final line.\n"
        "- Do not output \\boxed{}.\n"
        "- Do not add any text after the final line.\n\n"
        f"Problem:\n{question}\n\n"
        "Solution (end with the final line):\n"
    )

def build_retry_prompt(question: str, reflection: str, dataset: str = "") -> str:
    dataset = (dataset or "").lower()
    if dataset == "math":
        return (
            "Solve the problem again from scratch using the checklist.\n"
            "IMPORTANT:\n"
            "- Do NOT reuse or reference the previous solution.\n"
            "- Keep the solution concise.\n"
            "- Put ONLY the final answer on the last line as \\boxed{...}.\n"
            "- Do not write anything after the final line.\n\n"
            f"Checklist:\n{reflection}\n\n"
            f"Problem:\n{question}\n\n"
            "Solution:\n"
        )
    return (
        "You are a helpful math tutor. Use the reflection to solve correctly.\n"
        "IMPORTANT:\n"
        "- The FINAL line of your output must be exactly: #### <answer>\n"
        "- Do NOT write '####' anywhere except the final line.\n"
        "- Do not output \\boxed{}.\n"
        "- Do not add any text after the final line.\n\n"
        f"Reflection:\n{reflection}\n\n"
        f"Problem:\n{question}\n\n"
        "Solution (end with the final line):\n"
    )

def build_reflection_prompt_full(question: str, solution: str, pred_final: str) -> str:
    return (
        "You are analyzing a failed math solution to improve the next attempt.\n"
        "CRITICAL RULES:\n"
        "- Do NOT include the correct final answer.\n"
        "- Do NOT include ANY numbers or numerals (0-9) anywhere.\n"
        "- Output EXACTLY 3 lines in this format:\n"
        "ERROR_TYPE: <short>\n"
        "LIKELY_STEP: <short phrase or 'unknown'>\n"
        "FIX_PLAN: <one sentence>\n\n"
        f"Problem:\n{question}\n\n"
        f"Model's previous solution:\n{solution}\n\n"
        f"Model's parsed final answer: {pred_final}\n"
    )

def build_reflection_prompt_plan(question: str, pred_final: str) -> str:
    return (
        "You will solve the problem again. First, write a short plan/checklist.\n"
        "CRITICAL RULES:\n"
        "- Do NOT reference or quote any previous solution.\n"
        "- Do NOT include the correct final answer.\n"
        "- Do NOT include ANY numbers or numerals (0-9) anywhere.\n"
        "- Output EXACTLY 3 lines:\n"
        "ERROR_TYPE: <most likely mistake category>\n"
        "LIKELY_STEP: <short phrase or 'unknown'>\n"
        "FIX_PLAN: <one sentence checklist>\n\n"
        f"Problem:\n{question}\n\n"
        f"Model's parsed final answer (may be wrong): {pred_final}\n"
    )

# --- Gold extraction + formatting ---

def format_gold_final(dataset: str, ex: Dict[str, Any]) -> Optional[str]:
    dataset = (dataset or ex.get("dataset") or "").lower()
    ans = ex.get("answer", "")
    if not ans:
        return None
    if dataset == "gsm8k":
        gt = extract_final_answer_strict(ans)
        return f"#### {gt}".strip() if gt else None
    # MATH: assume processed data already stores final as a string number/expr.
    gt = str(ans).strip()
    if not gt:
        return None
    return f"\\boxed{{{gt}}}"

def synth_reflection_label(kind: str) -> str:
    # No digits allowed.
    if kind == "plan":
        return "ERROR_TYPE: arithmetic\nLIKELY_STEP: unknown\nFIX_PLAN: recheck algebra, constraints, and units"
    return "ERROR_TYPE: arithmetic\nLIKELY_STEP: unknown\nFIX_PLAN: recheck calculations and assumptions"

def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def make_example(prompt: str, target: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "metadata": meta,
    }

def infer_stage_from_mode(mode: str) -> str:
    m = (mode or "").lower()
    if m.endswith("_solve"):
        return "solve"
    if m.endswith("_label"):
        return "reflection"
    if m.endswith("_retry") or m == "retry_only":
        return "retry"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math_in", type=str, required=True)
    ap.add_argument("--gsm_in", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_total", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    math_ex = load_jsonl(args.math_in)
    gsm_ex = load_jsonl(args.gsm_in)

    # pool both datasets
    pool = []
    for ex in math_ex:
        pool.append(("math", ex))
    for ex in gsm_ex:
        pool.append(("gsm8k", ex))

    # Filter to those with q + gold
    filtered = []
    for ds, ex in pool:
        q = ex.get("question", "").strip()
        gt = format_gold_final(ds, ex)
        if q and gt:
            filtered.append((ds, q, gt))

    if not filtered:
        raise RuntimeError("No valid examples found. Check input paths + fields (question/answer).")

    # Build exactly 500 examples:
    # 100 solve (baseline), 100 retry_only, 100 reflect_full retry, 100 reflect_plan retry,
    # 50 reflection_full labels, 50 reflection_plan labels.
    plan = [
        ("baseline_solve", 100),
        ("retry_only", 100),
        ("reflect_full_retry", 100),
        ("reflect_plan_retry", 100),
        ("reflect_full_label", 50),
        ("reflect_plan_label", 50),
    ]
    assert sum(n for _, n in plan) == args.n_total, "Adjust counts or --n_total to match."

    out_rows: List[Dict[str, Any]] = []

    for mode, k in plan:
        for _ in range(k):
            ds, q, gt = rnd.choice(filtered)

            if mode == "baseline_solve":
                prompt = build_solve_prompt(q, ds)
                target = gt

            elif mode == "retry_only":
                prompt = build_retry_prompt(q, reflection="", dataset=ds)
                target = gt

            elif mode == "reflect_full_retry":
                reflection = synth_reflection_label(kind="full")
                prompt = build_retry_prompt(q, reflection=reflection, dataset=ds)
                target = gt

            elif mode == "reflect_plan_retry":
                reflection = synth_reflection_label(kind="plan")
                prompt = build_retry_prompt(q, reflection=reflection, dataset=ds)
                target = gt

            elif mode == "reflect_full_label":
                # Use placeholders; we only care about validating "no digits" loss masking.
                prompt = build_reflection_prompt_full(q, solution="<previous solution omitted>", pred_final="<previous answer omitted>")
                target = synth_reflection_label(kind="full")

            elif mode == "reflect_plan_label":
                prompt = build_reflection_prompt_plan(q, pred_final="<previous answer omitted>")
                target = synth_reflection_label(kind="plan")

            else:
                raise ValueError(mode)

            out_rows.append(make_example(
                prompt=prompt,
                target=target,
                meta={"dataset": ds, "mode": mode, "stage": infer_stage_from_mode(mode)},
            ))

    rnd.shuffle(out_rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} SFT examples -> {args.out}")
    modes = {}
    for r in out_rows:
        modes[r["metadata"]["mode"]] = modes.get(r["metadata"]["mode"], 0) + 1
    print("Counts by mode:", modes)

if __name__ == "__main__":
    main()
