import os
import json
from typing import Optional
import re
import time

from src.utils.answer_parser import (
    extract_final_answer_strict,
    extract_final_answer_loose,
    math_strict_equal,
)
from src.utils.generator import GenConfig, Generator
from fractions import Fraction

# --------- Helpers for MATH / normalization ---------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _to_fraction(s: str) -> Optional[Fraction]:
    if s is None:
        return None
    s = normalize_answer(s)
    if s is None:
        return None
    # handle simple fractions or integers/decimals
    try:
        if "/" in s:
            return Fraction(s)           # e.g. "5/6"
        # decimals -> Fraction exactly via string
        return Fraction(s)
    except Exception:
        return None


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
    return text[start: i - 1].strip()


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
    Parse model prediction final answer with sensible fallbacks.

    Returns:
      - strict: best-effort for STRICT format (e.g., "#### <ans>" for GSM8K)
      - loose: like strict, but will fall back only when strict is missing
              (guarantees loose is never "worse" than strict when strict exists)
      - boxed: last \boxed{...} (balanced braces) when present
      - source: where the final answer came from
    All fields are normalized with normalize_answer().
    """
    sol_text = sol_text or ""

    # Strip Qwen3 <think>...</think> blocks (8B thinking mode) before answer extraction.
    # Also strip unclosed <think> blocks in case the model never emitted </think>.
    sol_text = re.sub(r"<think>.*?</think>", "", sol_text, flags=re.DOTALL).strip()
    sol_text = re.sub(r"<think>.*$", "", sol_text, flags=re.DOTALL).strip()

    boxed = None
    if "\\boxed{" in sol_text:
        boxed = normalize_answer(extract_math_final(sol_text))

    strict = normalize_answer(extract_final_answer_strict(sol_text))
    # Prefer strict when available; otherwise relax.
    if strict is not None:
        loose = strict
        source = "strict"
    else:
        loose = normalize_answer(extract_final_answer_loose(sol_text))
        source = "loose" if loose is not None else None

        if loose is None and boxed is not None:
            loose = boxed
            source = "boxed"
        if loose is None:
            loose = normalize_answer(extract_math_final(sol_text))
            source = "last_line"

    # If strict missing, try boxed / last-line to populate strict too.
    if strict is None and boxed is not None:
        strict = boxed
    if strict is None:
        strict = normalize_answer(extract_math_final(sol_text))

    return {"strict": strict, "loose": loose, "boxed": boxed, "source": source}


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

# ---------------------------------------------------------------------------
# Few-shot examples for reflection prompts
# ---------------------------------------------------------------------------
# Rules demonstrated in every example:
#   - Exactly 3 output lines: ERROR_TYPE / LIKELY_STEP / FIX_PLAN
#   - Zero digits or numbers in the reflection output
#   - FIX_PLAN is one concrete, actionable sentence
# Two examples per mode: one GSM8K-style arithmetic, one MATH-style algebra/formula

_FEW_SHOT_FULL = """\
Here are two examples of correct reflections:

--- Example 1 ---
Problem:
A baker makes 24 muffins. She sells 15 at the morning market and 6 more in the afternoon. How many muffins are left?

Model's previous solution:
Morning sales: 15 muffins.
Afternoon sales: 6 muffins.
Remaining = 24 - 15 = 9 muffins.
#### 9

Model's parsed final answer: 9
ERROR_TYPE: incomplete subtraction
LIKELY_STEP: final subtraction
FIX_PLAN: Subtract the combined total of all sales sessions from the starting count, not just one session's sales.

--- Example 2 ---
Problem:
A train covers 120 miles at 60 mph, then 120 miles at 40 mph. What is the average speed for the whole journey?

Model's previous solution:
Speed on leg one = 60 mph, speed on leg two = 40 mph.
Average speed = (60 + 40) / 2 = 50 mph.
\\boxed{50}

Model's parsed final answer: 50
ERROR_TYPE: formula misapplication
LIKELY_STEP: average speed calculation
FIX_PLAN: Divide total distance by total elapsed time rather than taking the arithmetic mean of the two speeds.

--- End Examples ---
Now analyze this failed solution:
"""

_FEW_SHOT_PLAN = """\
Here are two examples of correct reflections:

--- Example 1 ---
Problem:
A baker makes 24 muffins. She sells 15 at the morning market and 6 more in the afternoon. How many muffins are left?

Model's parsed final answer (may be wrong): 9
ERROR_TYPE: incomplete subtraction
LIKELY_STEP: computing total items removed
FIX_PLAN: Identify every quantity that reduces the starting total, sum them, then subtract once from the initial count.

--- Example 2 ---
Problem:
A train covers 120 miles at 60 mph, then 120 miles at 40 mph. What is the average speed for the whole journey?

Model's parsed final answer (may be wrong): 50
ERROR_TYPE: formula misapplication
LIKELY_STEP: average speed calculation
FIX_PLAN: Compute the ratio of total distance to total travel time rather than averaging the individual leg speeds.

--- End Examples ---
Now analyze this problem:
"""

_FEW_SHOT_TAIL = """\
Here are two examples of correct reflections:

--- Example 1 ---
Problem:
A baker makes 24 muffins. She sells 15 at the morning market and 6 more in the afternoon. How many muffins are left?

Model's parsed final answer (may be wrong): 9
Last lines of model work (may be wrong):
Remaining = 24 - 15 = 9 muffins.
#### 9

ERROR_TYPE: incomplete subtraction
LIKELY_STEP: final subtraction step
FIX_PLAN: Account for all sales groups before computing the remainder, not just the first group.

--- Example 2 ---
Problem:
A train covers 120 miles at 60 mph, then 120 miles at 40 mph. What is the average speed?

Model's parsed final answer (may be wrong): 50
Last lines of model work (may be wrong):
Average speed = (60 + 40) / 2 = 50 mph.
\\boxed{50}

ERROR_TYPE: formula misapplication
LIKELY_STEP: final average calculation
FIX_PLAN: Replace the arithmetic mean of speeds with total distance divided by total travel time.

--- End Examples ---
Now diagnose this solution:
"""

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


def _fmt_pred_final(pred_final: Optional[str]) -> str:
    if pred_final is None:
        return "<unparsed>"
    s = str(pred_final).strip()
    return s if s else "<unparsed>"


def build_reflection_prompt_full(
    question: str,
    solution: str,
    pred_final: Optional[str],
    few_shot: bool = False,
) -> str:
    pf = _fmt_pred_final(pred_final)
    few_shot_block = (_FEW_SHOT_FULL + "\n") if few_shot else ""
    return (
        "You are analyzing a failed math solution to improve the next attempt.\n"
        "CRITICAL RULES:\n"
        "- Do NOT include the correct final answer.\n"
        "- Do NOT include ANY numbers at all (no digits, no decimals, no fractions).\n"
        "- Do NOT restate the full solution.\n"
        "Output exactly 3 lines in this format:\n"
        "ERROR_TYPE: <short>\n"
        "LIKELY_STEP: <step name or 'unknown'>\n"
        "FIX_PLAN: <one sentence>\n\n"
        + few_shot_block
        + f"Problem:\n{question}\n\n"
        f"Model's previous solution:\n{solution}\n\n"
        f"Model's parsed final answer: {pf}\n"
    )


def build_reflection_prompt_tail(
    question: str,
    solution: str,
    pred_final: Optional[str],
    few_shot: bool = False,
) -> str:
    pf = _fmt_pred_final(pred_final)
    tail = _tail_lines(solution, n_lines=6)
    few_shot_block = (_FEW_SHOT_TAIL + "\n") if few_shot else ""
    return (
        "You are diagnosing why the model's final answer is likely wrong.\n"
        "CRITICAL RULES:\n"
        "- Do NOT include the correct final answer.\n"
        "- Do NOT include ANY numbers at all (no digits, no decimals, no fractions).\n"
        "- Do NOT rewrite the full solution.\n"
        "Output exactly 3 lines in this format:\n"
        "ERROR_TYPE: <short>\n"
        "LIKELY_STEP: <step name or 'unknown'>\n"
        "FIX_PLAN: <one sentence checklist of what to verify>\n\n"
        + few_shot_block
        + f"Problem:\n{question}\n\n"
        f"Model's parsed final answer (may be wrong): {pf}\n\n"
        f"Last lines of model work (may be wrong):\n{tail}\n"
    )


def build_reflection_prompt_plan(
    question: str,
    pred_final: Optional[str],
    few_shot: bool = False,
) -> str:
    pf = _fmt_pred_final(pred_final)
    few_shot_block = (_FEW_SHOT_PLAN + "\n") if few_shot else ""
    return (
        "You will solve the problem again. First, write a short plan/checklist.\n"
        "CRITICAL RULES:\n"
        "- Do NOT reference or quote any previous solution.\n"
        "- Do NOT include the correct final answer.\n"
        "- Do NOT include ANY numbers at all (no digits, no decimals, no fractions).\n"
        "Output exactly 3 lines:\n"
        "ERROR_TYPE: <most likely mistake category>\n"
        "LIKELY_STEP: <where mistakes often happen, or 'unknown'>\n"
        "FIX_PLAN: <one sentence checklist of verifications>\n\n"
        + few_shot_block
        + f"Problem:\n{question}\n\n"
        f"Model's parsed final answer (may be wrong): {pf}\n"
    )


_GROUNDED_FEW_SHOT_EXAMPLE = """\
Here is a worked example of the format:

Problem:
A store sells pencils for $0.25 each. Jake buys 8 pencils. How much does he spend?

Failed attempt:
Cost per pencil: $0.25
Number of pencils: 8
Total cost: 8 × 0.25 = $1.80

Model's (wrong) answer: $1.80

WRONG LINE: "Total cost: 8 × 0.25 = $1.80"
WHY WRONG: 8 × 0.25 equals 2.00, not 1.80; the multiplication was computed incorrectly.
CORRECT VALUE: $2.00

---
Now do the same for the problem below.

"""


def build_reflection_prompt_grounded(
    question: str,
    solution: str,
    pred_final: Optional[str],
    few_shot: bool = False,
) -> str:
    """
    Grounded reflection prompt: forces the model to quote the exact wrong line
    from the failed attempt, name the specific error, and state the correct value.

    Unlike build_reflection_prompt_full, this:
    - Includes the full failed solution (not omitted)
    - Removes the "no digits" constraint — specific values are required
    - Uses a forced-quote format that can't be answered generically

    few_shot=True prepends a worked example demonstrating the desired output format.
    """
    pf = _fmt_pred_final(pred_final)
    few_shot_prefix = _GROUNDED_FEW_SHOT_EXAMPLE if few_shot else ""
    return (
        few_shot_prefix
        + "You are reviewing a failed math solution.\n"
        "The model's answer was wrong. Find the FIRST calculation mistake.\n\n"
        f"Problem:\n{question}\n\n"
        f"Failed attempt:\n{solution}\n\n"
        f"Model's (wrong) answer: {pf}\n\n"
        "Respond in EXACTLY this format (3 lines):\n"
        'WRONG LINE: "<copy the exact line from the failed attempt where the first error occurs>"\n'
        "WHY WRONG: <one sentence explaining the specific error in that line>\n"
        "CORRECT VALUE: <what the result of that line should be>\n\n"
        "Rules:\n"
        "- WRONG LINE must be a verbatim quote from the failed attempt above\n"
        "- Be specific about numbers, not generic (e.g. 'got 48 but should be 24')\n"
        "- Focus on the FIRST error that caused the wrong final answer\n"
    )


def build_retry_prompt_grounded(question: str, reflection: str, dataset: str = "") -> str:
    """
    Retry prompt for use with grounded reflections.
    Frames the reflection as a specific error correction rather than a checklist.
    """
    dataset = (dataset or "").lower()

    if dataset == "math":
        return (
            "Your previous attempt was wrong. Here is the specific mistake:\n\n"
            f"{reflection}\n\n"
            "Now solve the problem from scratch, avoiding that exact mistake.\n"
            "IMPORTANT:\n"
            "- Put ONLY the final answer on the last line as \\boxed{...}.\n"
            "- Do not write anything after the final line.\n\n"
            f"Problem:\n{question}\n\n"
            "Solution:\n"
        )

    # default: GSM8K-style
    return (
        "Your previous attempt was wrong. Here is the specific mistake:\n\n"
        f"{reflection}\n\n"
        "Now solve the problem from scratch, avoiding that exact mistake.\n"
        "IMPORTANT:\n"
        "- The FINAL line of your output must be exactly: #### <answer>\n"
        "- Do NOT write '####' anywhere except the final line.\n"
        "- Do not output \\boxed{}.\n"
        "- Do not add any text after the final line.\n\n"
        f"Problem:\n{question}\n\n"
        "Solution (end with the final line):\n"
    )


def build_reflection_prompt(
    mode: str,
    question: str,
    solution: str,
    pred_final: Optional[str],
    few_shot: bool = False,
) -> str:
    mode = (mode or "full").lower()
    if mode == "plan":
        return build_reflection_prompt_plan(question, pred_final, few_shot=few_shot)
    if mode == "tail":
        return build_reflection_prompt_tail(question, solution, pred_final, few_shot=few_shot)
    if mode == "grounded":
        return build_reflection_prompt_grounded(question, solution, pred_final, few_shot=few_shot)
    return build_reflection_prompt_full(question, solution, pred_final, few_shot=few_shot)



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


# --------- Robustness helpers ---------

def _count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _append_jsonl(f, obj: dict) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _checkpoint(f) -> None:
    f.flush()
    os.fsync(f.fileno())


# --------- NEW: Aggregate summary over an existing output jsonl ---------

def summarize_existing_output(output_jsonl: str) -> dict:
    """
    Recompute aggregate stats from an existing RRR output jsonl.

    - Keeps resume alignment: file may contain eval records + skipped/failed records.
    - Aggregate accuracy denominators use ONLY valid eval examples.
    """
    if not os.path.exists(output_jsonl):
        return {}

    # counts over ALL lines (including skipped/failed)
    num_total_lines = 0
    num_skipped = 0
    num_failed = 0

    # counts over VALID eval examples only
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

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            num_total_lines += 1

            try:
                rec = json.loads(line)
            except Exception:
                # treat unparsable lines as "failed-ish" for visibility
                num_failed += 1
                continue

            status = rec.get("status")
            if status == "skipped":
                num_skipped += 1
                continue
            if status == "failed":
                num_failed += 1
                continue

            # Otherwise: should be an eval record
            first = rec.get("first") or {}
            retry = rec.get("retry")  # may be None
            refl = rec.get("reflection")  # may be None

            n += 1
            first_correct_loose += int(bool(first.get("correct_loose")))
            first_correct_strict += int(bool(first.get("correct_strict")))

            if retry is not None:
                retries_attempted += 1
                retry_correct_loose += int(bool(retry.get("correct_loose")))
                retry_correct_strict += int(bool(retry.get("correct_strict")))

                if isinstance(refl, dict):
                    useful_reflections += int(bool(refl.get("useful_heuristic")))

            m1 = first.get("meta") or {}
            mr = (refl or {}).get("meta") if isinstance(refl, dict) else {}
            m2 = (retry or {}).get("meta") if isinstance(retry, dict) else {}

            t1 = int(m1.get("total_tokens", 0) or 0)
            tr = int((mr or {}).get("total_tokens", 0) or 0)
            t2 = int((m2 or {}).get("total_tokens", 0) or 0)

            l1 = float(m1.get("latency_s", 0.0) or 0.0)
            lr = float((mr or {}).get("latency_s", 0.0) or 0.0)
            l2 = float((m2 or {}).get("latency_s", 0.0) or 0.0)

            tokens_solve += t1
            tokens_reflect += tr
            tokens_retry += t2
            tokens_total += (t1 + tr + t2)

            latency_solve += l1
            latency_reflect += lr
            latency_retry += l2
            latency_total += (l1 + lr + l2)

    return {
        # file-level bookkeeping
        "num_total_lines": num_total_lines,
        "num_skipped": num_skipped,
        "num_failed": num_failed,

        # eval-level aggregates
        "n": n,
        "first_correct_loose": first_correct_loose,
        "first_correct_strict": first_correct_strict,
        "retry_correct_loose": retry_correct_loose,
        "retry_correct_strict": retry_correct_strict,
        "retries_attempted": retries_attempted,
        "useful_reflections": useful_reflections,
        "tokens_solve": tokens_solve,
        "tokens_reflect": tokens_reflect,
        "tokens_retry": tokens_retry,
        "tokens_total": tokens_total,
        "latency_total": latency_total,
        "latency_solve": latency_solve,
        "latency_reflect": latency_reflect,
        "latency_retry": latency_retry,
    }



# --------- Main eval ---------

def run_rrr_eval(
    gen,
    input_jsonl: str,
    output_jsonl: str,
    limit: int = None,
    solve_cfg=None,
    reflect_cfg=None,
    retry_cfg=None,
    no_reflect: bool = False,
    retry_only: bool = False,
    seed: int = 0,
    reflection_mode: str = "full",
    few_shot: bool = False,
    # NEW robustness knobs:
    resume: bool = False,
    checkpoint_every: int = 10,
    max_example_retries: int = 3,
    retry_backoff_sec: float = 2.0,
):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # Stats for THIS invocation
    n = 0  # total records written (valid + skipped + failed)
    n_valid = 0  # valid eval examples (denominator for accuracy)
    num_skipped = 0
    num_failed = 0
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

    # Determine resume offset (number of already-written records)
    done = 0
    out_mode = "w"
    if resume and os.path.exists(output_jsonl):
        done = _count_jsonl_lines(output_jsonl)
        out_mode = "a"
        print(
            f"[RRR] resume enabled: {output_jsonl} already has {done} lines; "
            f"skipping first {done} inputs and appending.",
            flush=True
        )

    # For progress printing: total remaining examples we *intend* to process
    if limit is None:
        with open(input_jsonl, "r", encoding="utf-8") as f_tmp:
            total_in_file = sum(1 for _ in f_tmp)
        total_examples = max(0, total_in_file - done)
    else:
        total_examples = max(0, limit - done)

    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, out_mode, encoding="utf-8") as f_out:
        written_since_start = 0

        for i, line in enumerate(f_in):
            if i < done:
                continue

            if limit is not None and i >= limit:
                break

            attempt = 0
            while True:
                try:
                    ex = json.loads(line)
                    dataset = ex.get("dataset", "")
                    q = ex.get("question", "")
                    gt_final = parse_gt_final(ex)

                    # If malformed, write a skip record so resume stays aligned
                    if not q or gt_final is None:
                        rec = {
                            "idx": i,
                            "status": "skipped",
                            "reason": "missing_question_or_gt",
                            "dataset": ex.get("dataset"),
                            "meta": ex.get("meta", {}),
                        }
                        _append_jsonl(f_out, rec)
                        n += 1
                        num_skipped += 1
                        written_since_start += 1
                        break

                    print(f"[RRR] Example {n_valid+1}/{max(total_examples,1)} (input_idx={i})", flush=True)

                    # 1) Solve (seeded)
                    sol1, meta1 = gen.generate(build_solve_prompt(q, dataset), solve_cfg, seed=seed + i)

                    t = int(meta1.get("total_tokens", 0))
                    tokens_solve += t
                    tokens_total += t

                    lt = float(meta1.get("latency_s", 0.0))
                    latency_solve += lt
                    latency_total += lt

                    p1 = parse_pred_final(sol1)
                    pred1_strict = p1["strict"]
                    pred1_loose = p1["loose"]
                    pred1_src = p1.get("source", "unknown")

                    ok1_strict = strict_match(ex, pred1_strict, gt_final)
                    ok1_loose  = loose_match(ex, pred1_loose, gt_final)

                    first_correct_strict += int(ok1_strict)
                    first_correct_loose += int(ok1_loose)

                    rec = {
                        "idx": i,
                        "question": q,
                        "dataset": ex.get("dataset"),
                        "meta": ex.get("meta", {}),
                        "gt_final": gt_final,
                        "seed": seed,
                        "first": {
                            "solution": sol1,
                            "pred_final_strict": pred1_strict,
                            "pred_final_loose": pred1_loose,
                            "pred_source": pred1_src,
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
                            print("[RRR]  -> wrong (loose), retry-only (no reflection)", flush=True)
                            refl_text = ""
                            useful = False
                            meta_r = {"skipped": True, "total_tokens": 0, "latency_s": 0.0}
                        else:
                            print(f"[RRR]  -> wrong (loose), reflecting (mode={reflection_mode})", flush=True)

                            # Use pred1_loose (the model's parsed answer) in the reflection prompt
                            refl_prompt = build_reflection_prompt(reflection_mode, q, sol1, pred1_loose, few_shot=few_shot)
                            refl_text, meta_r = gen.generate(
                                refl_prompt,
                                reflect_cfg,
                                seed=seed + i + 10_000,
                            )

                            t = int(meta_r.get("total_tokens", 0))
                            tokens_reflect += t
                            tokens_total += t

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

                        t = int(meta2.get("total_tokens", 0))
                        tokens_retry += t
                        tokens_total += t

                        lt = float(meta2.get("latency_s", 0.0))
                        latency_retry += lt
                        latency_total += lt

                        p2 = parse_pred_final(sol2)
                        pred2 = p2["loose"]
                        pred2_src = p2.get("source", "unknown")

                        ok2_strict = strict_match(ex, pred2, gt_final)
                        ok2_loose  = loose_match(ex, pred2, gt_final)

                        retry_correct_strict += int(ok2_strict)
                        retry_correct_loose += int(ok2_loose)

                        rec["reflection"] = {
                            "mode": reflection_mode,
                            "text": refl_text,
                            "useful_heuristic": useful,
                            "meta": meta_r,
                        }

                        rec["retry"] = {
                            "solution": sol2,
                            "pred_final": pred2,
                            "pred_source": pred2_src,
                            "correct_strict": ok2_strict,
                            "correct_loose": ok2_loose,
                            "meta": meta2,
                        }

                    _append_jsonl(f_out, rec)
                    n += 1
                    n_valid += 1
                    written_since_start += 1
                    break

                except Exception as e:
                    attempt += 1
                    if attempt > max_example_retries:
                        fail = {
                            "idx": i,
                            "status": "failed",
                            "error": repr(e),
                        }
                        _append_jsonl(f_out, fail)
                        n += 1
                        num_failed += 1
                        written_since_start += 1
                        print(f"[RRR][error] input_idx={i} failed permanently after {attempt-1} retries: {e}", flush=True)
                        break

                    sleep_s = retry_backoff_sec * (2 ** (attempt - 1))
                    print(f"[RRR][warn] input_idx={i} attempt={attempt} error={e} -> retrying in {sleep_s:.1f}s", flush=True)
                    time.sleep(sleep_s)

            if checkpoint_every and checkpoint_every > 0 and (written_since_start % checkpoint_every == 0):
                _checkpoint(f_out)

        _checkpoint(f_out)


    # ---- Print BOTH invocation stats and aggregate stats ----
    print(f"Wrote {n} new records in this invocation to {output_jsonl}")
    print(f"(Invocation) Skipped records: {num_skipped}")
    print(f"(Invocation) Failed records:  {num_failed}")
    print(f"(Invocation) Valid eval examples: {n_valid}")

    print(f"(Invocation) First-try accuracy (loose):  {first_correct_loose}/{max(n_valid,1)} = {first_correct_loose/max(n_valid,1):.3f}")
    print(f"(Invocation) First-try accuracy (strict): {first_correct_strict}/{max(n_valid,1)} = {first_correct_strict/max(n_valid,1):.3f}")
    if retries_attempted:
        print(f"(Invocation) Retry success (loose | conditional):  {retry_correct_loose}/{retries_attempted} = {retry_correct_loose/max(retries_attempted,1):.3f}")
        print(f"(Invocation) Retry success (strict| conditional): {retry_correct_strict}/{retries_attempted} = {retry_correct_strict/max(retries_attempted,1):.3f}")
        print(f"(Invocation) Overall accuracy (loose):  {(first_correct_loose+retry_correct_loose)}/{max(n_valid,1)} = {(first_correct_loose+retry_correct_loose)/max(n_valid,1):.3f}")
        print(f"(Invocation) Overall accuracy (strict): {(first_correct_strict+retry_correct_strict)}/{max(n_valid,1)} = {(first_correct_strict+retry_correct_strict)/max(n_valid,1):.3f}")

        if (not no_reflect) and (not retry_only):
            print(f"(Invocation) Reflection usefulness rate (heuristic): {useful_reflections}/{retries_attempted} = {useful_reflections/max(retries_attempted,1):.3f}")

    print(f"(Invocation) Tokens used (total): {tokens_total}")
    print(f"(Invocation) Avg tokens / example: {tokens_total/max(n_valid,1):.1f}")
    print(f"(Invocation) Tokens by stage: solve={tokens_solve}, reflect={tokens_reflect}, retry={tokens_retry}")

    print(f"(Invocation) Total latency (s): {latency_total:.2f}")
    print(f"(Invocation) Avg latency / example (s): {latency_total/max(n_valid,1):.3f}")
    print(f"(Invocation) Latency by stage: solve={latency_solve:.2f}, reflect={latency_reflect:.2f}, retry={latency_retry:.2f}")

    # Aggregate summary over full file (so rerun prints real totals)
    agg = summarize_existing_output(output_jsonl) if resume else {}
    if agg:
        print("\n[RRR] Aggregate summary from file:")
        print(f"Total lines in file: {agg.get('num_total_lines', 0)}")
        print(f"Skipped records:     {agg.get('num_skipped', 0)}")
        print(f"Failed records:      {agg.get('num_failed', 0)}")

        N = agg.get("n", 0)
        if N <= 0:
            print("No valid eval records found (only skipped/failed).")
        else:
            print(f"Valid eval examples: {N}")
            print(f"First-try accuracy (loose):  {agg['first_correct_loose']}/{N} = {agg['first_correct_loose']/max(N,1):.3f}")
            print(f"First-try accuracy (strict): {agg['first_correct_strict']}/{N} = {agg['first_correct_strict']/max(N,1):.3f}")

            if agg["retries_attempted"]:
                ra = agg["retries_attempted"]
                print(f"Retry success (loose | conditional):  {agg['retry_correct_loose']}/{ra} = {agg['retry_correct_loose']/max(ra,1):.3f}")
                print(f"Retry success (strict| conditional): {agg['retry_correct_strict']}/{ra} = {agg['retry_correct_strict']/max(ra,1):.3f}")
                print(f"Overall accuracy (loose):  {(agg['first_correct_loose']+agg['retry_correct_loose'])}/{N} = {(agg['first_correct_loose']+agg['retry_correct_loose'])/max(N,1):.3f}")
                print(f"Overall accuracy (strict): {(agg['first_correct_strict']+agg['retry_correct_strict'])}/{N} = {(agg['first_correct_strict']+agg['retry_correct_strict'])/max(N,1):.3f}")
                print(f"Reflection usefulness rate (heuristic): {agg['useful_reflections']}/{ra} = {agg['useful_reflections']/max(ra,1):.3f}")

            print(f"Tokens used (total): {agg['tokens_total']}")
            print(f"Avg tokens / example: {agg['tokens_total']/max(N,1):.1f}")
            print(f"Tokens by stage: solve={agg['tokens_solve']}, reflect={agg['tokens_reflect']}, retry={agg['tokens_retry']}")
            print(f"Total latency (s): {agg['latency_total']:.2f}")
            print(f"Avg latency / example (s): {agg['latency_total']/max(N,1):.3f}")
            print(f"Latency by stage: solve={agg['latency_solve']:.2f}, reflect={agg['latency_reflect']:.2f}, retry={agg['latency_retry']:.2f}")

