import os
import json
from typing import Optional
import re

from src.utils.answer_parser import (
    extract_final_answer_strict,
    extract_final_answer_loose,
    math_strict_equal,
)
from src.utils.generator import GenConfig, Generator

# --------- Helpers for MATH / normalization ---------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_last_boxed_balanced(text: str) -> Optional[str]:
    """
    Return content of the last \boxed{...} with balanced braces.
    """
    if not text:
        return None
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    i = idx + len(r"\boxed{")
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None
    return text[start : i - 1].strip()


def extract_math_final(text: str) -> Optional[str]:
    """
    Extract last \boxed{...} if present (balanced braces), else last non-empty line.
    """
    if not text:
        return None

    boxed = extract_last_boxed_balanced(text)
    if boxed:
        return boxed

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """
    Light normalization to avoid format mismatches:
    - strip whitespace
    - remove enclosing $...$
    - remove commas in numbers (1,000 -> 1000)
    - collapse spaces
    """
    if ans is None:
        return None
    s = str(ans).strip()
    if not s:
        return None

    # Strip common LaTeX math delimiters
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    # Remove commas inside numbers
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else None


def parse_gt_final(ex: dict) -> Optional[str]:
    """
    Dataset-aware gold final answer extraction.
    - GSM8K: gold 'answer' contains #### delimiter.
    - MATH: processed data should have final answer in ex['answer'] already,
            but we also support boxed/solution-like strings.
    """
    dataset = ex.get("dataset", "")
    gt_sol = ex.get("answer", "")

    if dataset == "gsm8k":
        gt = extract_final_answer_strict(gt_sol)  # expects ####
        return normalize_answer(gt)

    if dataset == "math":
        gt = normalize_answer(gt_sol)
        if gt is None:
            return None
        if "\\boxed{" in gt_sol:
            gt2 = extract_math_final(gt_sol)
            return normalize_answer(gt2)
        return gt

    # Fallback
    gt = extract_final_answer_strict(gt_sol)
    if gt is not None:
        return normalize_answer(gt)
    gt2 = extract_math_final(gt_sol)
    if gt2 is not None:
        return normalize_answer(gt2)
    return normalize_answer(gt_sol)


def parse_pred_final(sol_text: str) -> dict:
    """
    Parse model prediction final answer with fallbacks.
    Returns dict with strict/loose/boxed values (all normalized).
    """
    strict = normalize_answer(extract_final_answer_strict(sol_text))
    loose = normalize_answer(extract_final_answer_loose(sol_text))

    boxed = None
    if "\\boxed{" in (sol_text or ""):
        boxed = normalize_answer(extract_math_final(sol_text))

    # fallback to boxed / last line
    if strict is None and boxed is not None:
        strict = boxed
    if loose is None and boxed is not None:
        loose = boxed

    if strict is None:
        strict = normalize_answer(extract_math_final(sol_text))
    if loose is None:
        loose = normalize_answer(extract_math_final(sol_text))

    return {"strict": strict, "loose": loose, "boxed": boxed}


def strict_match(ex: dict, pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    dataset = (ex.get("dataset") or "").lower()
    if dataset == "math":
        return math_strict_equal(pred, gt)
    return normalize_answer(pred) == normalize_answer(gt)


def loose_match(ex: dict, pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    dataset = (ex.get("dataset") or "").lower()
    if dataset == "math":
        return math_strict_equal(pred, gt)
    return normalize_answer(pred) == normalize_answer(gt)

# --------- Prompts ---------

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

    # default: GSM8K-style
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


def _tail_lines(text: str, n_lines: int = 6) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-n_lines:]).strip()


def build_reflection_prompt_full(
    question: str,
    solution: str,
    pred_final: Optional[str],
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
    )


def build_reflection_prompt_tail(
    question: str,
    solution: str,
    pred_final: Optional[str],
) -> str:
    tail = _tail_lines(solution, n_lines=6)
    return (
        "You are diagnosing why the model's final answer is likely wrong.\n"
        "DO NOT include the correct final answer or any numeric final answer.\n"
        "Do NOT rewrite the full solution.\n"
        "Output exactly 3 lines in this format:\n"
        "ERROR_TYPE: <short>\n"
        "LIKELY_STEP: <step number or 'unknown'>\n"
        "FIX_PLAN: <one sentence checklist of what to verify>\n\n"
        f"Problem:\n{question}\n\n"
        f"Model's parsed final answer (may be wrong): {pred_final}\n\n"
        f"Last lines of model work (may be wrong):\n{tail}\n"
    )


def build_reflection_prompt_plan(
    question: str,
    pred_final: Optional[str],
) -> str:
    return (
        "You will solve the problem again. First, write a short plan/checklist.\n"
        "Rules:\n"
        "- Do NOT reference or quote any previous solution.\n"
        "- Do NOT include the correct final answer or any numeric final answer.\n"
        "- Output exactly 3 lines:\n"
        "ERROR_TYPE: <most likely mistake category>\n"
        "LIKELY_STEP: <where mistakes often happen, or 'unknown'>\n"
        "FIX_PLAN: <one sentence checklist of verifications>\n\n"
        f"Problem:\n{question}\n\n"
        f"Model's parsed final answer (may be wrong): {pred_final}\n"
    )


def build_reflection_prompt(
    mode: str,
    question: str,
    solution: str,
    pred_final: Optional[str],
) -> str:
    mode = (mode or "full").lower()
    if mode == "plan":
        return build_reflection_prompt_plan(question, pred_final)
    if mode == "tail":
        return build_reflection_prompt_tail(question, solution, pred_final)
    return build_reflection_prompt_full(question, solution, pred_final)


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

    # default: GSM8K-style
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


# --------- Reflection usefulness heuristic ---------

_REFLECT_KEYWORDS = [
    "recheck", "verify", "recompute", "substitute", "simplify", "factor",
    "cases", "domain", "constraint", "units", "re-derive", "derive", "check",
    "plug", "plug in", "compute again", "arithmetic", "algebra"
]

def reflection_useful_heuristic(reflection_text: str) -> bool:
    t = (reflection_text or "").lower()
    return any(k in t for k in _REFLECT_KEYWORDS)

# --------- Main eval ---------

def run_rrr_eval(
    gen: Generator,
    input_jsonl: str,
    output_jsonl: str,
    limit: int = 50,
    solve_cfg: GenConfig = GenConfig(max_new_tokens=256, temperature=0.7, top_p=0.95),
    reflect_cfg: GenConfig = GenConfig(max_new_tokens=128, temperature=0.3, top_p=0.9),
    retry_cfg: GenConfig = GenConfig(max_new_tokens=256, temperature=0.7, top_p=0.95),
    no_reflect: bool = False,
    retry_only: bool = False,
    seed: int = 0,
    reflection_mode: str = "full",
):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    n = 0
    first_correct_loose = 0
    first_correct_strict = 0
    retry_correct_loose = 0
    retry_correct_strict = 0
    retries_attempted = 0

    useful_reflections = 0

    tokens_solve = 0
    tokens_reflect = 0
    tokens_retry = 0
    tokens_total = 0

    latency_total = 0.0
    latency_solve = 0.0
    latency_reflect = 0.0
    latency_retry = 0.0

    # For nicer progress printing: know how many examples we'll run
    if limit is None:
        with open(input_jsonl, "r", encoding="utf-8") as f_tmp:
            total_examples = sum(1 for _ in f_tmp)
    else:
        total_examples = limit

    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if limit is not None and i >= limit:
                break

            ex = json.loads(line)
            dataset = ex.get("dataset", "")
            q = ex.get("question", "")
            gt_final = parse_gt_final(ex)

            if not q or gt_final is None:
                continue

            print(f"[RRR] Example {n+1}/{total_examples}", flush=True)

            # 1) Solve (seeded)
            sol1, meta1 = gen.generate(build_solve_prompt(q, dataset), solve_cfg, seed=seed + i)

            # tokens
            t = int(meta1.get("total_tokens", 0))
            tokens_solve += t
            tokens_total += t

            # latency
            lt = float(meta1.get("latency_s", 0.0))
            latency_solve += lt
            latency_total += lt

            p1 = parse_pred_final(sol1)
            pred1_strict = p1["strict"]
            pred1_loose = p1["loose"]

            ok1_strict = strict_match(ex, pred1_strict, gt_final)
            ok1_loose = loose_match(ex, pred1_loose, gt_final)

            first_correct_strict += int(ok1_strict)
            first_correct_loose += int(ok1_loose)

            rec = {
                "question": q,
                "dataset": ex.get("dataset"),
                "meta": ex.get("meta", {}),
                "gt_final": gt_final,
                "seed": seed,
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

            # Reflect + retry based on LOOSE correctness
            if (not ok1_loose) and (not no_reflect):
                retries_attempted += 1

                if retry_only:
                    # --- RETRY-ONLY: no reflection generation ---
                    print("[RRR]  -> wrong (loose), retry-only (no reflection)", flush=True)
                    refl_text = ""
                    useful = False
                    meta_r = {"skipped": True, "total_tokens": 0, "latency_s": 0.0}
                else:
                    # --- NORMAL: generate reflection (mode-controlled) ---
                    print(f"[RRR]  -> wrong (loose), reflecting (mode={reflection_mode})", flush=True)

                    refl_prompt = build_reflection_prompt(reflection_mode, q, sol1, pred1_loose)
                    refl_text, meta_r = gen.generate(
                        refl_prompt,
                        reflect_cfg,
                        seed=seed + i + 10_000,
                    )

                    # tokens
                    t = int(meta_r.get("total_tokens", 0))
                    tokens_reflect += t
                    tokens_total += t

                    # latency
                    lt = float(meta_r.get("latency_s", 0.0))
                    latency_reflect += lt
                    latency_total += lt

                    refl_text = _first_3_lines(refl_text)
                    useful = reflection_useful_heuristic(refl_text)
                    useful_reflections += int(useful)

                print("[RRR]  -> retrying", flush=True)

                sol2, meta2 = gen.generate(
                    build_retry_prompt(q, refl_text, dataset),
                    retry_cfg,
                    seed=seed + i + 20_000,
                )

                # tokens
                t = int(meta2.get("total_tokens", 0))
                tokens_retry += t
                tokens_total += t

                # latency
                lt = float(meta2.get("latency_s", 0.0))
                latency_retry += lt
                latency_total += lt

                p2 = parse_pred_final(sol2)
                pred2_strict = p2["strict"]
                pred2_loose = p2["loose"]

                ok2_strict = strict_match(ex, pred2_strict, gt_final)
                ok2_loose = loose_match(ex, pred2_loose, gt_final)

                retry_correct_strict += int(ok2_strict)
                retry_correct_loose += int(ok2_loose)

                # Record reflection info (skipped vs generated)
                rec["reflection"] = {
                    "mode": reflection_mode,
                    "text": refl_text,
                    "useful_heuristic": useful,
                    "meta": meta_r,
                }

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

        if (not no_reflect) and (not retry_only):
            print(f"Reflection usefulness rate (heuristic): {useful_reflections}/{retries_attempted} = {useful_reflections/max(retries_attempted,1):.3f}")

    print(f"Tokens used (total): {tokens_total}")
    print(f"Avg tokens / example: {tokens_total/max(n,1):.1f}")
    print(f"Tokens by stage: solve={tokens_solve}, reflect={tokens_reflect}, retry={tokens_retry}")

    print(f"Total latency (s): {latency_total:.2f}")
    print(f"Avg latency / example (s): {latency_total/max(n,1):.3f}")
    print(f"Latency by stage: solve={latency_solve:.2f}, reflect={latency_reflect:.2f}, retry={latency_retry:.2f}")
