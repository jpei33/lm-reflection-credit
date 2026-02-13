import os
import json
from dataclasses import dataclass
from typing import Optional

from src.utils.answer_parser import extract_final_answer_strict, extract_final_answer_loose
from src.utils.generator import GenConfig, Generator


def build_solve_prompt(question: str) -> str:
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


def build_reflection_prompt(
    question: str,
    solution: str,
    pred_final: Optional[str],
    gt_final: str,
) -> str:
    return (
        "You are analyzing a failed math solution to improve the next attempt.\n"
        "DO NOT include the correct final answer or any numeric final answer.\n"
        "Output exactly 3 lines in this format:\n"
        "ERROR_TYPE: <short>\n"
        "LIKELY_STEP: <step number or 'unknown'>\n"
        "FIX_PLAN: <one sentence>\n\n"
        f"Problem:\n{question}\n\n"
        f"Model's previous solution:\n{solution}\n\n"
        f"Model's parsed final answer: {pred_final}\n"
        f"Correct final answer: {gt_final}\n"
    )


def build_retry_prompt(question: str, reflection: str) -> str:
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


def _first_3_lines(text: str) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[:3]).strip()


def run_rrr_eval(
    gen: Generator,
    input_jsonl: str,
    output_jsonl: str,
    limit: int = 50,
    solve_cfg: GenConfig = GenConfig(max_new_tokens=256, temperature=0.7, top_p=0.95),
    reflect_cfg: GenConfig = GenConfig(max_new_tokens=128, temperature=0.3, top_p=0.9),
    retry_cfg: GenConfig = GenConfig(max_new_tokens=256, temperature=0.7, top_p=0.95),
):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    n = 0

    # We'll report "loose" accuracy as primary (more realistic),
    # and keep "strict" as an ablation metric.
    first_correct_loose = 0
    first_correct_strict = 0

    retry_correct_loose = 0
    retry_correct_strict = 0
    retries_attempted = 0

    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= limit:
                break

            ex = json.loads(line)
            q = ex["question"]
            gt_sol = ex["answer"]
            gt_final = extract_final_answer_strict(gt_sol)  # GSM8K answers have #### in the gold
            if gt_final is None:
                # skip weird example
                continue

            print(f"[RRR] Example {i+1}/{limit}", flush=True)

            # 1) Solve
            sol1, meta1 = gen.generate(build_solve_prompt(q), solve_cfg)

            pred1_strict = extract_final_answer_strict(sol1)
            pred1_loose = extract_final_answer_loose(sol1)

            ok1_strict = (pred1_strict == gt_final)
            ok1_loose = (pred1_loose == gt_final)

            first_correct_strict += int(ok1_strict)
            first_correct_loose += int(ok1_loose)

            rec = {
                "question": q,
                "gt_final": gt_final,
                "first": {
                    "solution": sol1,
                    "pred_final_strict": pred1_strict,
                    "pred_final_loose": pred1_loose,
                    "correct_strict": ok1_strict,
                    "correct_loose": ok1_loose,
                    "meta": meta1,
                },
                "reflection": None,
                "retry": None,
            }

            # Decide whether to reflect+retry based on LOOSE correctness (practical)
            if not ok1_loose:
                retries_attempted += 1
                print("[RRR]  ↳ wrong (loose), reflecting", flush=True)

                refl_text, meta_r = gen.generate(
                    build_reflection_prompt(q, sol1, pred1_loose, gt_final),
                    reflect_cfg,
                )
                refl_text = _first_3_lines(refl_text)

                print("[RRR]  ↳ retrying", flush=True)
                sol2, meta2 = gen.generate(build_retry_prompt(q, refl_text), retry_cfg)

                pred2_strict = extract_final_answer_strict(sol2)
                pred2_loose = extract_final_answer_loose(sol2)

                ok2_strict = (pred2_strict == gt_final)
                ok2_loose = (pred2_loose == gt_final)

                retry_correct_strict += int(ok2_strict)
                retry_correct_loose += int(ok2_loose)

                rec["reflection"] = {"text": refl_text, "meta": meta_r}
                rec["retry"] = {
                    "solution": sol2,
                    "pred_final_strict": pred2_strict,
                    "pred_final_loose": pred2_loose,
                    "correct_strict": ok2_strict,
                    "correct_loose": ok2_loose,
                    "meta": meta2,
                }

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples to {output_jsonl}")
    print(f"First-try accuracy (loose):  {first_correct_loose}/{n} = {first_correct_loose/max(n,1):.3f}")
    print(f"First-try accuracy (strict): {first_correct_strict}/{n} = {first_correct_strict/max(n,1):.3f}")
    if retries_attempted:
        print(f"Retry success (loose | conditional):  {retry_correct_loose}/{retries_attempted} = {retry_correct_loose/max(retries_attempted,1):.3f}")
        print(f"Retry success (strict| conditional): {retry_correct_strict}/{retries_attempted} = {retry_correct_strict/max(retries_attempted,1):.3f}")
        print(f"Overall accuracy (loose):  {(first_correct_loose+retry_correct_loose)}/{n} = {(first_correct_loose+retry_correct_loose)/max(n,1):.3f}")
        print(f"Overall accuracy (strict): {(first_correct_strict+retry_correct_strict)}/{n} = {(first_correct_strict+retry_correct_strict)/max(n,1):.3f}")
