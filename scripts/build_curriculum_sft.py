"""
build_curriculum_sft.py
-----------------------
Build full-scale SFT training datasets for all 4 model conditions,
with difficulty metadata and curriculum ordering.

Outputs (in data/processed/curriculum/):
  baseline_solve.jsonl       -- direct solve (gold answer as target)
  retry_only.jsonl           -- retry with empty reflection
  reflect_full_retry.jsonl   -- plan-label + retry (full-reflection mode)
  reflect_plan_retry.jsonl   -- plan-label + retry (plan-reflection mode)
  all_modes_curriculum.jsonl -- all 4 combined, sorted by difficulty

Each example carries:
  metadata.difficulty    int 1-5  (curriculum key; lower = easier)
  metadata.step_count    int      (raw GSM8K calc steps, or MATH level as int)
  metadata.level         str      (MATH "Level N" or GSM8K difficulty band)
  metadata.dataset       str      "math" | "gsm8k"
  metadata.mode          str      one of the 4 mode keys
  metadata.stage         str      "solve" | "reflection" | "retry"
  metadata.subject       str      MATH subject or "" for GSM8K

Difficulty scoring:
  MATH   -- "Level 1"..."Level 5" parsed directly -> 1...5
  GSM8K  -- count <<...>> calculator annotations in the gold solution,
             then bin into quintiles across the train split -> 1...5

Curriculum training recipe (recommended):
  Sort each dataset by difficulty ascending and train in order
  (or use staged epochs: 1 epoch easy, 1 epoch medium+hard).

Usage:
  python scripts/build_curriculum_sft.py \\
      --math_in  data/processed/math_train.jsonl \\
      --gsm_in   data/processed/gsm8k_train.jsonl \\
      --out_dir  data/processed/curriculum \\
      --seed     42

Windows PowerShell:
  python scripts\\build_curriculum_sft.py `
      --math_in  data\\processed\\math_train.jsonl `
      --gsm_in   data\\processed\\gsm8k_train.jsonl `
      --out_dir  data\\processed\\curriculum `
      --seed     42
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt builders  (mirrors build_tiny_sft_dataset.py exactly)
# ---------------------------------------------------------------------------

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


def synth_reflection_label(kind: str) -> str:
    """Synthetic reflection target (no digits allowed)."""
    if kind == "plan":
        return (
            "ERROR_TYPE: arithmetic\n"
            "LIKELY_STEP: unknown\n"
            "FIX_PLAN: recheck algebra, constraints, and units"
        )
    return (
        "ERROR_TYPE: arithmetic\n"
        "LIKELY_STEP: unknown\n"
        "FIX_PLAN: recheck calculations and assumptions"
    )


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

_CALC_ANNOTATION_RE = re.compile(r"<<[^>]+>>")


def gsm8k_step_count(answer_str: str) -> int:
    """Count <<...>> calculator annotations in the GSM8K gold answer string."""
    return len(_CALC_ANNOTATION_RE.findall(answer_str or ""))


def math_level_to_int(level_str: Optional[str]) -> int:
    """Convert 'Level N' -> int 1-5.  Returns 3 (medium) if unparseable."""
    if not level_str:
        return 3
    m = re.search(r"(\d+)", level_str)
    if m:
        return max(1, min(5, int(m.group(1))))
    return 3


def assign_difficulty_gsm8k(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute step_count (<<...>> annotation count) for every GSM8K example,
    then map to difficulty 1-5 using fixed thresholds derived from the
    full-split distribution (min=0, max=9, median=3):
      0-1 steps  -> difficulty 1 (very easy)
      2 steps    -> difficulty 2
      3 steps    -> difficulty 3
      4 steps    -> difficulty 4
      5+ steps   -> difficulty 5 (hard)
    This produces all 5 distinct difficulty levels and is interpretable.
    """
    def _bin(c: int) -> int:
        if c <= 1:  return 1
        if c == 2:  return 2
        if c == 3:  return 3
        if c == 4:  return 4
        return 5

    out = []
    for ex in examples:
        ex = dict(ex)
        cnt = gsm8k_step_count(ex.get("answer", ""))
        ex["_step_count"] = cnt
        ex["_difficulty"] = _bin(cnt)
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Gold answer formatting
# ---------------------------------------------------------------------------

def _extract_gsm8k_final(answer_str: str) -> Optional[str]:
    """Return the numeric answer from a GSM8K answer string (after ####)."""
    if not answer_str:
        return None
    # GSM8K answers end with "#### <number>"
    m = re.search(r"####\s*(.+)$", answer_str.strip(), re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback: last line
    lines = [ln.strip() for ln in answer_str.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def format_gold_target(dataset: str, ex: Dict[str, Any]) -> Optional[str]:
    """Return the gold target string in the correct format for training."""
    dataset = (dataset or ex.get("dataset") or "").lower()
    if dataset == "gsm8k":
        final = _extract_gsm8k_final(ex.get("answer", ""))
        return f"#### {final}" if final else None
    # MATH
    ans = str(ex.get("answer", "")).strip()
    return f"\\boxed{{{ans}}}" if ans else None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):>6,} examples -> {path}")


# ---------------------------------------------------------------------------
# Example builder
# ---------------------------------------------------------------------------

def make_example(
    prompt: str,
    target: str,
    mode: str,
    stage: str,
    dataset: str,
    difficulty: int,
    step_count: int,
    level: str,
    subject: str,
    question_id: str,
) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "id":         question_id,
            "dataset":    dataset,
            "mode":       mode,
            "stage":      stage,
            "difficulty": difficulty,   # 1 (easy) ... 5 (hard) -- curriculum key
            "step_count": step_count,   # raw step count (GSM8K) or MATH level int
            "level":      level,        # "Level N" (MATH) or "GSM8K-D{n}" (GSM8K)
            "subject":    subject,      # MATH subject or "" for GSM8K
        },
    }


# ---------------------------------------------------------------------------
# Per-mode dataset builders
# ---------------------------------------------------------------------------

def build_baseline_solve(
    examples: List[Tuple[str, Dict[str, Any], str, int, int, str, str, str]]
) -> List[Dict[str, Any]]:
    """Mode: baseline_solve — direct solve, gold answer as target."""
    out = []
    for ds, ex, gold, diff, steps, level, subject, qid in examples:
        prompt = build_solve_prompt(ex["question"], ds)
        out.append(make_example(
            prompt=prompt, target=gold, mode="baseline_solve", stage="solve",
            dataset=ds, difficulty=diff, step_count=steps,
            level=level, subject=subject, question_id=qid,
        ))
    return out


def build_retry_only(
    examples: List[Tuple[str, Dict[str, Any], str, int, int, str, str, str]]
) -> List[Dict[str, Any]]:
    """Mode: retry_only — retry prompt with empty reflection, gold answer as target."""
    out = []
    for ds, ex, gold, diff, steps, level, subject, qid in examples:
        prompt = build_retry_prompt(ex["question"], reflection="", dataset=ds)
        out.append(make_example(
            prompt=prompt, target=gold, mode="retry_only", stage="retry",
            dataset=ds, difficulty=diff, step_count=steps,
            level=level, subject=subject, question_id=qid,
        ))
    return out


def build_reflect_full_retry(
    examples: List[Tuple[str, Dict[str, Any], str, int, int, str, str, str]]
) -> List[Dict[str, Any]]:
    """
    Mode: reflect_full_retry -- two turns per example:
      1. reflection-generation turn (synthetic label, no digits)
      2. retry turn (gold answer)
    """
    out = []
    for ds, ex, gold, diff, steps, level, subject, qid in examples:
        q = ex["question"]
        refl_label = synth_reflection_label(kind="full")

        # Turn 1: teach the model to generate a full reflection
        refl_prompt = build_reflection_prompt_full(
            q,
            solution="<previous solution omitted>",
            pred_final="<previous answer omitted>",
        )
        out.append(make_example(
            prompt=refl_prompt, target=refl_label,
            mode="reflect_full_retry", stage="reflection",
            dataset=ds, difficulty=diff, step_count=steps,
            level=level, subject=subject, question_id=qid + "_refl",
        ))

        # Turn 2: retry with the reflection
        retry_prompt = build_retry_prompt(q, reflection=refl_label, dataset=ds)
        out.append(make_example(
            prompt=retry_prompt, target=gold,
            mode="reflect_full_retry", stage="retry",
            dataset=ds, difficulty=diff, step_count=steps,
            level=level, subject=subject, question_id=qid + "_retry",
        ))
    return out


def build_reflect_plan_retry(
    examples: List[Tuple[str, Dict[str, Any], str, int, int, str, str, str]]
) -> List[Dict[str, Any]]:
    """
    Mode: reflect_plan_retry -- two turns per example:
      1. plan-reflection turn (synthetic label, no digits)
      2. retry turn (gold answer)
    """
    out = []
    for ds, ex, gold, diff, steps, level, subject, qid in examples:
        q = ex["question"]
        refl_label = synth_reflection_label(kind="plan")

        # Turn 1: plan reflection
        refl_prompt = build_reflection_prompt_plan(
            q,
            pred_final="<previous answer omitted>",
        )
        out.append(make_example(
            prompt=refl_prompt, target=refl_label,
            mode="reflect_plan_retry", stage="reflection",
            dataset=ds, difficulty=diff, step_count=steps,
            level=level, subject=subject, question_id=qid + "_refl",
        ))

        # Turn 2: retry with plan
        retry_prompt = build_retry_prompt(q, reflection=refl_label, dataset=ds)
        out.append(make_example(
            prompt=retry_prompt, target=gold,
            mode="reflect_plan_retry", stage="retry",
            dataset=ds, difficulty=diff, step_count=steps,
            level=level, subject=subject, question_id=qid + "_retry",
        ))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build full-scale curriculum SFT datasets for all 4 model conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--math_in",  default="data/processed/math_train.jsonl",
                    help="Processed MATH train JSONL (from prepare_data.py)")
    ap.add_argument("--gsm_in",   default="data/processed/gsm8k_train.jsonl",
                    help="Processed GSM8K train JSONL (from prepare_data.py)")
    ap.add_argument("--out_dir",  default="data/processed/curriculum",
                    help="Output directory for curriculum datasets")
    ap.add_argument("--seed",     type=int, default=42,
                    help="Random seed for shuffling within difficulty tiers")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Cap total examples per source (for quick smoke tests)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    print(f"\n[build_curriculum_sft] Loading data...")
    math_raw = load_jsonl(args.math_in)
    gsm_raw  = load_jsonl(args.gsm_in)

    if args.limit:
        math_raw = math_raw[:args.limit]
        gsm_raw  = gsm_raw[:args.limit]

    print(f"  MATH train: {len(math_raw):,} examples")
    print(f"  GSM8K train: {len(gsm_raw):,} examples")

    # ── Difficulty scoring ──────────────────────────────────────────────────
    print("\n[build_curriculum_sft] Scoring difficulty...")

    # MATH: Level 1-5 directly from meta.level
    math_scored = []
    for ex in math_raw:
        level_str = (ex.get("meta") or {}).get("level") or ""
        diff = math_level_to_int(level_str)
        subject = (ex.get("meta") or {}).get("type") or ""
        math_scored.append((
            "math", ex,
            format_gold_target("math", ex),
            diff,
            diff,           # step_count = level int for MATH
            level_str or f"Level {diff}",
            subject,
            ex.get("id", f"math_{len(math_scored):06d}"),
        ))
    # Drop examples with no gold answer
    math_scored = [t for t in math_scored if t[2]]

    # GSM8K: step_count proxy -> quintile difficulty
    gsm_with_steps = assign_difficulty_gsm8k(gsm_raw)
    gsm_scored = []
    for ex in gsm_with_steps:
        diff   = ex["_difficulty"]
        steps  = ex["_step_count"]
        level  = f"GSM8K-D{diff}"
        gold   = format_gold_target("gsm8k", ex)
        if not gold:
            continue
        gsm_scored.append((
            "gsm8k", ex, gold,
            diff, steps, level, "",
            ex.get("id", f"gsm8k_{len(gsm_scored):06d}"),
        ))

    print(f"  MATH valid: {len(math_scored):,}  |  GSM8K valid: {len(gsm_scored):,}")

    # Print difficulty distributions
    for name, scored in [("MATH", math_scored), ("GSM8K", gsm_scored)]:
        from collections import Counter
        dist = Counter(t[3] for t in scored)
        print(f"  {name} difficulty dist: { {k: dist[k] for k in sorted(dist)} }")

    # ── Combine and curriculum-sort helper ──────────────────────────────────
    all_examples = math_scored + gsm_scored

    def curriculum_sort(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by difficulty ASC, shuffle within each difficulty tier."""
        tiers: Dict[int, List] = {i: [] for i in range(1, 6)}
        for r in rows:
            tiers[r["metadata"]["difficulty"]].append(r)
        out = []
        for d in range(1, 6):
            tier = tiers[d]
            rng.shuffle(tier)
            out.extend(tier)
        return out

    # ── Build each mode dataset ─────────────────────────────────────────────
    print("\n[build_curriculum_sft] Building mode datasets...")

    modes = {
        "baseline_solve":     build_baseline_solve(all_examples),
        "retry_only":         build_retry_only(all_examples),
        "reflect_full_retry": build_reflect_full_retry(all_examples),
        "reflect_plan_retry": build_reflect_plan_retry(all_examples),
    }

    all_rows: List[Dict[str, Any]] = []

    for mode_name, rows in modes.items():
        rows = curriculum_sort(rows)
        write_jsonl(out_dir / f"{mode_name}.jsonl", rows)
        all_rows.extend(rows)

    # ── Combined curriculum file ────────────────────────────────────────────
    # Sort all modes together by difficulty, shuffle within tier
    all_rows = curriculum_sort(all_rows)
    write_jsonl(out_dir / "all_modes_curriculum.jsonl", all_rows)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n[build_curriculum_sft] Done. Output: {out_dir}/")
    print(f"  Files written:")
    for mode_name in list(modes.keys()) + ["all_modes_curriculum"]:
        p = out_dir / f"{mode_name}.jsonl"
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            print(f"    {mode_name}.jsonl   ({n:,} examples)")

    print(f"\n  Curriculum metadata fields on every example:")
    print(f"    metadata.difficulty  int 1-5  (1=easy, 5=hard)")
    print(f"    metadata.step_count  int      (GSM8K calc steps | MATH level int)")
    print(f"    metadata.level       str      ('Level N' | 'GSM8K-D{{n}}')")
    print(f"    metadata.subject     str      (MATH subject | '' for GSM8K)")
    print(f"    metadata.dataset     str      ('math' | 'gsm8k')")
    print(f"    metadata.mode        str      (one of 4 mode keys)")
    print(f"    metadata.stage       str      ('solve' | 'reflection' | 'retry')")


if __name__ == "__main__":
    main()
