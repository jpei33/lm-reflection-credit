"""
compare_results.py
------------------
Compare eval results across all training phases:
  Phase 1 → SFT (behavioral cloning on 4 reasoning formats)
  Phase 2 → RLVR (rejection-sampling fine-tuning, current RFT approach)
  Phase 3 → GRPO (group-relative policy optimization, full paper method)

Each phase column is shown only when result files exist, so the same
script works at every stage of the experiment.

Usage:
  python scripts\\compare_results.py                        # auto-discover all
  python scripts\\compare_results.py --dataset gsm8k
  python scripts\\compare_results.py --dataset both --rank 8 --seed 42
  python scripts\\compare_results.py --phase sft            # SFT only
  python scripts\\compare_results.py --phase rlvr           # RLVR only
  python scripts\\compare_results.py --phase grpo           # GRPO only

Auto-discovery looks for JSONL files in results/runs/ matching each
condition's known run-name pattern.
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------
# Each condition maps to (sft_run_template, rlvr_run_template, grpo_run_template).
# Templates use {rank} and {seed} placeholders.
# None means no run exists for that phase/condition.

_CONDITIONS = [
    {
        "label":  "Baseline CoT",
        "sft":    "qwen3-4b-baseline_solve-r{rank}-seed{seed}",
        "rlvr":   "baseline_rlvr_solve-r{rank}-seed{seed}",
        "grpo":   "baseline_grpo_solve-r{rank}-seed{seed}",
        "eval_mode": "baseline_solve",
    },
    {
        "label":  "Retry Only",
        "sft":    "qwen3-4b-retry_only-r{rank}-seed{seed}",
        "rlvr":   "baseline_rlvr_retry-r{rank}-seed{seed}",
        "grpo":   "baseline_grpo_retry-r{rank}-seed{seed}",
        "eval_mode": "retry_only",
    },
    {
        "label":  "Reflect-Full + Retry",
        "sft":    "qwen3-4b-reflect_full_retry-r{rank}-seed{seed}",
        "rlvr":   "rrr-full-r{rank}-seed{seed}",
        "grpo":   "rrr_grpo-full-r{rank}-seed{seed}",
        "eval_mode": "reflect_full_retry",
    },
    {
        "label":  "Reflect-Plan + Retry",
        "sft":    "qwen3-4b-reflect_plan_retry-r{rank}-seed{seed}",
        "rlvr":   "rrr-plan-r{rank}-seed{seed}",
        "grpo":   "rrr_grpo-plan-r{rank}-seed{seed}",
        "eval_mode": "reflect_plan_retry",
    },
    {
        "label":  "Step Credit",
        "sft":    None,  # no SFT-only baseline for step-credit
        "rlvr":   "step_credit-r{rank}-seed{seed}",
        "grpo":   None,  # not implemented in Phase 3
        "eval_mode": "baseline_solve",
    },
]

_PHASES = ["sft", "rlvr", "grpo"]

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results" / "runs"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") in ("skipped", "failed"):
                    continue
                rows.append(r)
            except Exception:
                pass
    return rows


def resolve_run_name(template: Optional[str], rank: int, seed: int) -> Optional[str]:
    if template is None:
        return None
    return template.format(rank=rank, seed=seed)


def find_file(results_dir: Path, run_name: Optional[str], dataset: str) -> Optional[Path]:
    if run_name is None:
        return None
    p = results_dir / f"{run_name}_{dataset}.jsonl"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def first_acc(rows: List[dict]) -> float:
    if not rows:
        return 0.0
    return sum(bool(r.get("first", {}).get("correct_loose")) for r in rows) / len(rows)


def system_acc(rows: List[dict]) -> float:
    """First-try OR retry correct (system-level accuracy)."""
    if not rows:
        return 0.0
    correct = 0
    for r in rows:
        if r.get("first", {}).get("correct_loose"):
            correct += 1
        elif r.get("retry") and r["retry"].get("correct_loose"):
            correct += 1
    return correct / len(rows)


def avg_tokens(rows: List[dict]) -> float:
    if not rows:
        return 0.0
    total = 0
    for r in rows:
        total += r.get("first", {}).get("meta", {}).get("total_tokens", 0)
        if r.get("reflection") and isinstance(r["reflection"], dict):
            total += r["reflection"].get("meta", {}).get("total_tokens", 0)
        if r.get("retry") and isinstance(r["retry"], dict):
            total += r["retry"].get("meta", {}).get("total_tokens", 0)
    return total / len(rows)


def difficulty_breakdown(rows: List[dict]) -> Dict[str, Tuple[float, int]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        lvl = str((r.get("meta") or {}).get("level", "?"))
        buckets[lvl].append(r)
    result = {}
    for lvl in sorted(buckets, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)):
        result[lvl] = (system_acc(buckets[lvl]), len(buckets[lvl]))
    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_acc(val: Optional[float], std: Optional[float] = None) -> str:
    if val is None:
        return "    n/a   "
    if std is not None:
        return f"{val:.1%}±{std:.1%}"
    return f"{val:>6.1%}"


def _fmt_delta(val: Optional[float]) -> str:
    if val is None:
        return "   n/a "
    return f"{val:>+7.1%}"


# ---------------------------------------------------------------------------
# Multi-seed aggregation helpers
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) for a list of floats. std=0 if only one value."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _agg_system_acc(rows_per_seed: List[Optional[List[dict]]]) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean±std system accuracy across seeds. Returns (mean, std) or (None, None)."""
    accs = [system_acc(r) for r in rows_per_seed if r]
    if not accs:
        return None, None
    mean, std = _mean_std(accs)
    return mean, std


def _agg_first_acc(rows_per_seed: List[Optional[List[dict]]]) -> Tuple[Optional[float], Optional[float]]:
    accs = [first_acc(r) for r in rows_per_seed if r]
    if not accs:
        return None, None
    return _mean_std(accs)


# ---------------------------------------------------------------------------
# Best-of-N file helpers
# ---------------------------------------------------------------------------

def find_bon_file(results_dir: Path, run_name: str, n: int, dataset: str) -> Optional[Path]:
    p = results_dir / f"{run_name}_bon_n{n}_{dataset}.jsonl"
    return p if p.exists() else None


def bon_acc(rows: List[dict]) -> float:
    """Accuracy from BoN result file (any_correct field)."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("any_correct", False)) / len(rows)


# ---------------------------------------------------------------------------
# Core report
# ---------------------------------------------------------------------------

def report(
    phase_data: Dict[str, Dict[str, Optional[List[dict]]]],
    dataset: str,
    phases_present: List[str],
    multi_seed_data: Optional[Dict[str, Dict[str, List[Optional[List[dict]]]]]] = None,
    bon_data: Optional[Dict[int, Dict[str, float]]] = None,  # {N: {label: acc}}
) -> None:
    """
    phase_data:       phase_data[phase][label] = rows (single seed, primary)
    multi_seed_data:  multi_seed_data[phase][label] = [rows_seed0, rows_seed1, ...]
    bon_data:         bon_data[N][label] = accuracy
    """
    n_examples = 0
    for phase_rows in phase_data.values():
        for rows in phase_rows.values():
            if rows:
                n_examples = len(rows)
                break

    use_multi_seed = bool(multi_seed_data)

    print(f"\n{'='*80}")
    print(f"  DATASET: {dataset.upper()}   ({n_examples} examples per condition)")
    if use_multi_seed:
        print(f"  (values shown as mean ± std across seeds)")
    print(f"{'='*80}")

    # ── Overall accuracy table ───────────────────────────────────────────────
    col_w = 14 if use_multi_seed else 8
    hdr = f"  {'Condition':<24}"
    if "sft"  in phases_present: hdr += f"{'SFT':>{col_w}}"
    if "rlvr" in phases_present: hdr += f"{'RLVR':>{col_w}}  {'Δ(RLVR)':>8}"
    if "grpo" in phases_present: hdr += f"{'GRPO':>{col_w}}  {'Δ(GRPO)':>8}"
    if bon_data:
        for n_val in sorted(bon_data.keys()):
            hdr += f"  {'BoN-'+str(n_val):>8}"
    hdr += f"  {'Tokens':>7}"
    print(f"\n  System accuracy (first-try OR retry correct):\n")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cond in _CONDITIONS:
        label = cond["label"]
        rows_by_phase = {ph: phase_data.get(ph, {}).get(label) for ph in _PHASES}

        if use_multi_seed:
            sft_mean,  sft_std  = _agg_system_acc(multi_seed_data.get("sft",  {}).get(label, []))
            rlvr_mean, rlvr_std = _agg_system_acc(multi_seed_data.get("rlvr", {}).get(label, []))
            grpo_mean, grpo_std = _agg_system_acc(multi_seed_data.get("grpo", {}).get(label, []))
            sft_acc, rlvr_acc, grpo_acc = sft_mean, rlvr_mean, grpo_mean
        else:
            sft_acc  = system_acc(rows_by_phase["sft"])  if rows_by_phase["sft"]  else None
            rlvr_acc = system_acc(rows_by_phase["rlvr"]) if rows_by_phase["rlvr"] else None
            grpo_acc = system_acc(rows_by_phase["grpo"]) if rows_by_phase["grpo"] else None
            sft_std = rlvr_std = grpo_std = None

        rlvr_delta = (rlvr_acc - sft_acc) if (rlvr_acc is not None and sft_acc is not None) else None
        grpo_delta = (grpo_acc - sft_acc) if (grpo_acc is not None and sft_acc is not None) else None

        tok_rows = rows_by_phase.get("grpo") or rows_by_phase.get("rlvr") or rows_by_phase.get("sft") or []
        tok = avg_tokens(tok_rows) if tok_rows else None

        row = f"  {label:<24}"
        if "sft"  in phases_present:
            row += f"  {_fmt_acc(sft_acc, sft_std if use_multi_seed else None):>{col_w}}"
        if "rlvr" in phases_present:
            row += f"  {_fmt_acc(rlvr_acc, rlvr_std if use_multi_seed else None):>{col_w}}  {_fmt_delta(rlvr_delta):>8}"
        if "grpo" in phases_present:
            row += f"  {_fmt_acc(grpo_acc, grpo_std if use_multi_seed else None):>{col_w}}  {_fmt_delta(grpo_delta):>8}"
        if bon_data:
            for n_val in sorted(bon_data.keys()):
                bon_acc_val = bon_data[n_val].get(label)
                row += f"  {_fmt_acc(bon_acc_val):>8}"
        row += f"  {tok:>7.0f}" if tok else "       n/a"
        print(row)

    # ── Best-of-N section ────────────────────────────────────────────────────
    if bon_data:
        print(f"\n  Best-of-N vs trained models (compute-matched check):")
        print(f"  If RLVR > BoN-N, training adds real value beyond sampling more.")
        for cond in _CONDITIONS:
            label = cond["label"]
            rlvr_rows = phase_data.get("rlvr", {}).get(label)
            if not rlvr_rows:
                continue
            rlvr_a = system_acc(rlvr_rows)
            parts = [f"{label:<26}: RLVR={rlvr_a:.1%}"]
            for n_val in sorted(bon_data.keys()):
                bon_a = bon_data[n_val].get(label)
                if bon_a is not None:
                    delta = rlvr_a - bon_a
                    parts.append(f"BoN-{n_val}={bon_a:.1%}(Δ{delta:+.1%})")
            print(f"    {'  '.join(parts)}")

    # ── RLVR lift bar chart ───────────────────────────────────────────────────
    if "sft" in phases_present and "rlvr" in phases_present:
        print(f"\n  RLVR lift over SFT (system accuracy):")
        for cond in _CONDITIONS:
            label = cond["label"]
            sft_rows  = phase_data.get("sft",  {}).get(label)
            rlvr_rows = phase_data.get("rlvr", {}).get(label)
            if not sft_rows or not rlvr_rows:
                continue
            delta = system_acc(rlvr_rows) - system_acc(sft_rows)
            bar_filled = round(abs(delta) * 30)
            bar = ("+" if delta >= 0 else "-") * bar_filled
            print(f"    {label:<26}: {delta:>+7.1%}  {bar}")

    if "sft" in phases_present and "grpo" in phases_present:
        print(f"\n  GRPO lift over SFT (system accuracy):")
        for cond in _CONDITIONS:
            label = cond["label"]
            sft_rows  = phase_data.get("sft",  {}).get(label)
            grpo_rows = phase_data.get("grpo", {}).get(label)
            if not sft_rows or not grpo_rows:
                continue
            delta = system_acc(grpo_rows) - system_acc(sft_rows)
            bar_filled = round(abs(delta) * 30)
            bar = ("+" if delta >= 0 else "-") * bar_filled
            print(f"    {label:<26}: {delta:>+7.1%}  {bar}")

    if "rlvr" in phases_present and "grpo" in phases_present:
        print(f"\n  GRPO lift over RLVR (system accuracy):")
        for cond in _CONDITIONS:
            label = cond["label"]
            rlvr_rows = phase_data.get("rlvr", {}).get(label)
            grpo_rows = phase_data.get("grpo", {}).get(label)
            if not rlvr_rows or not grpo_rows:
                continue
            delta = system_acc(grpo_rows) - system_acc(rlvr_rows)
            bar_filled = round(abs(delta) * 30)
            bar = ("+" if delta >= 0 else "-") * bar_filled
            print(f"    {label:<26}: {delta:>+7.1%}  {bar}")

    # ── First-try accuracy breakdown ─────────────────────────────────────────
    print(f"\n  First-try accuracy (solve only, no retry):\n")
    hdr2 = f"  {'Condition':<24}"
    if "sft"  in phases_present: hdr2 += f"{'SFT':>{col_w}}"
    if "rlvr" in phases_present: hdr2 += f"{'RLVR':>{col_w}}  {'Δ(RLVR)':>8}"
    if "grpo" in phases_present: hdr2 += f"{'GRPO':>{col_w}}  {'Δ(GRPO)':>8}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for cond in _CONDITIONS:
        label = cond["label"]
        rows_by_phase = {ph: phase_data.get(ph, {}).get(label) for ph in _PHASES}

        if use_multi_seed:
            sft_fa,  sft_fstd  = _agg_first_acc(multi_seed_data.get("sft",  {}).get(label, []))
            rlvr_fa, rlvr_fstd = _agg_first_acc(multi_seed_data.get("rlvr", {}).get(label, []))
            grpo_fa, grpo_fstd = _agg_first_acc(multi_seed_data.get("grpo", {}).get(label, []))
        else:
            sft_fa  = first_acc(rows_by_phase["sft"])  if rows_by_phase["sft"]  else None
            rlvr_fa = first_acc(rows_by_phase["rlvr"]) if rows_by_phase["rlvr"] else None
            grpo_fa = first_acc(rows_by_phase["grpo"]) if rows_by_phase["grpo"] else None
            sft_fstd = rlvr_fstd = grpo_fstd = None

        rlvr_d = (rlvr_fa - sft_fa) if (rlvr_fa is not None and sft_fa is not None) else None
        grpo_d = (grpo_fa - sft_fa) if (grpo_fa is not None and sft_fa is not None) else None

        row = f"  {label:<24}"
        if "sft"  in phases_present:
            row += f"  {_fmt_acc(sft_fa, sft_fstd if use_multi_seed else None):>{col_w}}"
        if "rlvr" in phases_present:
            row += f"  {_fmt_acc(rlvr_fa, rlvr_fstd if use_multi_seed else None):>{col_w}}  {_fmt_delta(rlvr_d):>8}"
        if "grpo" in phases_present:
            row += f"  {_fmt_acc(grpo_fa, grpo_fstd if use_multi_seed else None):>{col_w}}  {_fmt_delta(grpo_d):>8}"
        print(row)

    # ── Difficulty breakdown ─────────────────────────────────────────────────
    print(f"\n  System accuracy by difficulty level (best available phase per condition):")

    all_levels: set = set()
    for phase_rows in phase_data.values():
        for rows in phase_rows.values():
            if rows:
                for lvl in difficulty_breakdown(rows):
                    all_levels.add(lvl)

    if all_levels:
        lvl_sorted = sorted(all_levels, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
        col_w2 = 9
        hdr3 = f"  {'Condition':<24}" + "".join(f"{'L'+l:>{col_w2}}" for l in lvl_sorted)
        print(hdr3)
        print("  " + "-" * (len(hdr3) - 2))

        for cond in _CONDITIONS:
            label = cond["label"]
            rows_by_phase = {ph: phase_data.get(ph, {}).get(label) for ph in _PHASES}
            rows = rows_by_phase.get("grpo") or rows_by_phase.get("rlvr") or rows_by_phase.get("sft")
            if not rows:
                continue
            bd = difficulty_breakdown(rows)
            row_str = f"  {label:<24}"
            for lvl in lvl_sorted:
                if lvl in bd:
                    acc, n = bd[lvl]
                    row_str += f"{acc:>{col_w2}.1%}"
                else:
                    row_str += f"{'N/A':>{col_w2}}"
            print(row_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare SFT / RLVR / GRPO eval results across all training conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset",  default="both", choices=["gsm8k", "math", "both"])
    ap.add_argument("--rank",     type=int, default=8)
    ap.add_argument("--phase",    default="all",
                    choices=["all", "sft", "rlvr", "grpo"],
                    help="Which phase(s) to include. 'all' auto-detects available files.")
    ap.add_argument("--results_dir", default=str(_RESULTS_DIR))

    # ── Multi-seed ──────────────────────────────────────────────────────────
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to aggregate across. Single seed = no error bars. "
                         "E.g. --seeds 42 0 1")

    # ── Best-of-N ───────────────────────────────────────────────────────────
    ap.add_argument("--bon_n", type=int, nargs="*", default=None,
                    help="Best-of-N values to include if eval_best_of_n.py has been run. "
                         "E.g. --bon_n 1 2 3 8.  Auto-detects if omitted.")
    ap.add_argument("--bon_run_name", default=None,
                    help="Run name of the SFT checkpoint used for BoN eval. "
                         "Defaults to qwen3-4b-baseline_solve-r{rank}-seed{primary_seed}.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    primary_seed = args.seeds[0]
    multi_seed   = len(args.seeds) > 1

    datasets = ["gsm8k", "math"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        # ── Primary seed phase data (used for delta calculations + BoN) ─────
        phase_data: Dict[str, Dict[str, Optional[List[dict]]]] = {
            "sft": {}, "rlvr": {}, "grpo": {}
        }

        # ── Multi-seed data: phase → label → [rows_per_seed] ─────────────────
        ms_data: Dict[str, Dict[str, List[Optional[List[dict]]]]] = {
            "sft": defaultdict(list),
            "rlvr": defaultdict(list),
            "grpo": defaultdict(list),
        }

        print(f"\n[load] Scanning {results_dir} for dataset={ds} ...")

        for cond in _CONDITIONS:
            label = cond["label"]
            for phase in _PHASES:
                if args.phase != "all" and args.phase != phase:
                    continue
                tmpl = cond.get(phase)

                for seed in args.seeds:
                    run_name = resolve_run_name(tmpl, args.rank, seed)
                    fpath = find_file(results_dir, run_name, ds)
                    if fpath:
                        rows = load_jsonl(fpath)
                        ms_data[phase][label].append(rows)
                        if seed == primary_seed:
                            phase_data[phase][label] = rows
                            print(f"  [load] [{phase:4s}] seed={seed} {label:<26} {len(rows):>4} rows")
                    else:
                        ms_data[phase][label].append(None)
                        if seed == primary_seed:
                            phase_data[phase][label] = None
                            if run_name:
                                print(f"  [miss] [{phase:4s}] seed={seed} {label:<26} ({run_name}_{ds}.jsonl)")

        # ── Best-of-N loading ────────────────────────────────────────────────
        bon_run = args.bon_run_name or f"qwen3-4b-baseline_solve-r{args.rank}-seed{primary_seed}"

        # Auto-detect N values if not specified
        if args.bon_n is None:
            detected_n = []
            for n_try in [1, 2, 3, 4, 8, 16]:
                p = results_dir / f"{bon_run}_bon_n{n_try}_{ds}.jsonl"
                if p.exists():
                    detected_n.append(n_try)
            bon_n_values = detected_n
        else:
            bon_n_values = args.bon_n

        bon_data: Dict[int, Dict[str, float]] = {}
        for n_val in bon_n_values:
            fpath = results_dir / f"{bon_run}_bon_n{n_val}_{ds}.jsonl"
            if fpath.exists():
                rows = load_jsonl(fpath)
                # BoN acc applies to baseline condition (same SFT checkpoint)
                acc = bon_acc(rows)
                bon_data[n_val] = {"Baseline CoT": acc}
                print(f"  [load] [bon ] N={n_val:<2} Baseline CoT  {len(rows):>4} rows  acc={acc:.1%}")

        # ── Determine which phases have any data ──────────────────────────────
        phases_present = [
            ph for ph in _PHASES
            if args.phase in ("all", ph) and any(v is not None for v in phase_data[ph].values())
        ]

        if not phases_present:
            print(f"\n[skip] No result files found for dataset={ds}.")
            print(f"       Run eval_sft.py for each condition first.")
            continue

        report(
            phase_data=phase_data,
            dataset=ds,
            phases_present=phases_present,
            multi_seed_data=ms_data if multi_seed else None,
            bon_data=bon_data if bon_data else None,
        )

    print()


if __name__ == "__main__":
    main()
