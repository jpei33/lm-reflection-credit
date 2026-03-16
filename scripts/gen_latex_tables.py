"""
scripts/gen_latex_tables.py
============================
Generate publication-ready LaTeX tables from experiment results.

Produces three tables:

  Table 1 — Main results: SFT / RLVR / GRPO across all 4B conditions
  Table 2 — Scaling: 4B vs 8B RLVR and GRPO (Retry Only + Reflect-Full)
  Table 3 — Inference baselines: pass@k and majority@k for Baseline CoT

Usage
-----
  python scripts/gen_latex_tables.py               # prints all tables to stdout
  python scripts/gen_latex_tables.py --out tables.tex   # writes to file
  python scripts/gen_latex_tables.py --table 1     # single table

Tables use:
  - \\pm for mean ± std notation
  - \\textbf{} for column-best values
  - \\dag for results statistically within noise of best (within 1 std)
  - booktabs rules (\\toprule, \\midrule, \\bottomrule)
  - siunitx S columns for aligned decimal points
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS      = _REPO_ROOT / "results" / "runs"

SEEDS = [42, 0, 1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _system_acc_loose(path: Path) -> Optional[float]:
    """Return system accuracy (first OR retry correct_loose) for one eval file."""
    if not path.exists():
        return None
    correct, total = 0, 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if (r.get("first") or {}).get("correct_loose"):
                correct += 1
            elif (r.get("retry") or {}).get("correct_loose"):
                correct += 1
    return correct / total if total else None


def _agg(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, std) in percent, ignoring None entries."""
    vals = [v * 100 for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def load_condition(run_pattern: str, mode_suffix: str) -> Dict[str, Tuple]:
    """
    Load mean±std for GSM8K and MATH across SEEDS.
    run_pattern: e.g. "rrr-full-r8-seed{s}"  (use {s} as seed placeholder)
    mode_suffix: e.g. "reflect_full_retry"
    Returns {"gsm8k": (mean, std), "math": (mean, std)}
    """
    gsm_vals, math_vals = [], []
    for s in SEEDS:
        run = run_pattern.format(s=s)
        gsm_path  = _RUNS / f"{run}_{mode_suffix}_gsm8k.jsonl"
        math_path = _RUNS / f"{run}_{mode_suffix}_math.jsonl"
        gsm_vals.append(_system_acc_loose(gsm_path))
        math_vals.append(_system_acc_loose(math_path))
    return {
        "gsm8k": _agg(gsm_vals),
        "math":  _agg(math_vals),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_cell(mean: Optional[float], std: Optional[float],
             bold: bool = False, dag: bool = False) -> str:
    """Format a (mean, std) pair as a LaTeX cell string."""
    if mean is None:
        return "---"
    std_str = f" \\pm {std:.1f}" if std is not None and std > 0.0 else ""
    inner   = f"{mean:.1f}{std_str}"
    if bold:
        inner = f"\\textbf{{{inner}}}"
    if dag:
        inner = inner + "$^\\dag$"
    return inner


def best_in_col(cells: List[Tuple[Optional[float], Optional[float]]]) -> int:
    """Return index of column-best (highest) mean, ignoring None."""
    best_idx, best_val = -1, -1e9
    for i, (m, _) in enumerate(cells):
        if m is not None and m > best_val:
            best_val = m
            best_idx = i
    return best_idx


def within_noise(mean: Optional[float], std: Optional[float],
                 best_mean: Optional[float]) -> bool:
    """True if mean is within 1 std of the best mean (i.e. not significantly worse)."""
    if mean is None or best_mean is None or std is None:
        return False
    return (best_mean - mean) <= std


# ---------------------------------------------------------------------------
# Table 1 — Main 4B results
# ---------------------------------------------------------------------------

def table1() -> str:
    """
    Main results table: SFT / RLVR / GRPO for all 4B conditions.
    Rows: Baseline CoT, Retry Only, Reflect-Full+Retry, Reflect-Plan+Retry, Step Credit
    """
    conditions = [
        # (label,                   sft_run,                        rlvr_run,                        grpo_run,                         mode_suffix)
        ("Baseline CoT",      "qwen3-4b-baseline_solve-r8-seed{s}",   "baseline_rlvr_solve-r8-seed{s}",  "baseline_grpo_solve-r8-seed{s}",  "baseline_solve"),
        ("Retry Only",         "qwen3-4b-retry_only-r8-seed{s}",       "baseline_rlvr_retry-r8-seed{s}",  "baseline_grpo_retry-r8-seed{s}",  "retry_only"),
        ("Reflect-Full+Retry", "qwen3-4b-reflect_full_retry-r8-seed{s}","rrr-full-r8-seed{s}",            "rrr_grpo-full-r8-seed{s}",        "reflect_full_retry"),
        ("Reflect-Plan+Retry", "qwen3-4b-reflect_plan_retry-r8-seed{s}","rrr-plan-r8-seed{s}",            None,                              "reflect_plan_retry"),
        ("Step Credit",        "qwen3-4b-step_credit-r8-seed{s}",       "step_credit-r8-seed{s}",          None,                              "retry_only"),
    ]

    # Collect all data
    data = []
    for label, sft_pat, rlvr_pat, grpo_pat, mode in conditions:
        sft_res  = load_condition(sft_pat,  mode) if sft_pat  else {"gsm8k": (None,None), "math": (None,None)}
        rlvr_res = load_condition(rlvr_pat, mode) if rlvr_pat else {"gsm8k": (None,None), "math": (None,None)}
        grpo_res = load_condition(grpo_pat, mode) if grpo_pat else {"gsm8k": (None,None), "math": (None,None)}
        data.append((label, sft_res, rlvr_res, grpo_res))

    # Find column-best for bolding
    def col_vals(key, res_idx):
        return [row[res_idx][key] for row in data]

    best_sft_gsm  = best_in_col(col_vals("gsm8k", 1))
    best_sft_math = best_in_col(col_vals("math",  1))
    best_rlvr_gsm  = best_in_col(col_vals("gsm8k", 2))
    best_rlvr_math = best_in_col(col_vals("math",  2))
    best_grpo_gsm  = best_in_col(col_vals("gsm8k", 3))
    best_grpo_math = best_in_col(col_vals("math",  3))

    # Best values for dag detection
    def get_best_mean(col_vals_list, best_idx):
        if best_idx < 0: return None
        return col_vals_list[best_idx][0]

    lines = []
    lines.append(r"% Table 1: Main results (4B model, mean ± std across 3 seeds)")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{System accuracy (first-try OR retry correct) on GSM8K and MATH")
    lines.append(r"           (200 examples each), mean $\pm$ std across 3 random seeds.")
    lines.append(r"           \textbf{Bold} = column best; $^\dag$ = within 1\,std of best.")
    lines.append(r"           All models: Qwen3-4B-Instruct-2507, LoRA rank\,8.}")
    lines.append(r"  \label{tab:main_results}")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \begin{tabular}{lcccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{2}{c}{\textbf{SFT}} & \multicolumn{2}{c}{\textbf{RLVR}} & \multicolumn{2}{c}{\textbf{GRPO}} \\")
    lines.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"    \textbf{Condition} & GSM8K & MATH & GSM8K & MATH & GSM8K & MATH \\")
    lines.append(r"    \midrule")

    for i, (label, sft_res, rlvr_res, grpo_res) in enumerate(data):
        sft_g_m,  sft_g_s  = sft_res["gsm8k"]
        sft_m_m,  sft_m_s  = sft_res["math"]
        rlvr_g_m, rlvr_g_s = rlvr_res["gsm8k"]
        rlvr_m_m, rlvr_m_s = rlvr_res["math"]
        grpo_g_m, grpo_g_s = grpo_res["gsm8k"]
        grpo_m_m, grpo_m_s = grpo_res["math"]

        sft_g_best_mean  = get_best_mean(col_vals("gsm8k", 1), best_sft_gsm)
        sft_m_best_mean  = get_best_mean(col_vals("math",  1), best_sft_math)
        rlvr_g_best_mean = get_best_mean(col_vals("gsm8k", 2), best_rlvr_gsm)
        rlvr_m_best_mean = get_best_mean(col_vals("math",  2), best_rlvr_math)
        grpo_g_best_mean = get_best_mean(col_vals("gsm8k", 3), best_grpo_gsm)
        grpo_m_best_mean = get_best_mean(col_vals("math",  3), best_grpo_math)

        c1 = fmt_cell(sft_g_m,  sft_g_s,  bold=(i==best_sft_gsm),
                      dag=(i!=best_sft_gsm  and within_noise(sft_g_m,  sft_g_s,  sft_g_best_mean)))
        c2 = fmt_cell(sft_m_m,  sft_m_s,  bold=(i==best_sft_math),
                      dag=(i!=best_sft_math and within_noise(sft_m_m,  sft_m_s,  sft_m_best_mean)))
        c3 = fmt_cell(rlvr_g_m, rlvr_g_s, bold=(i==best_rlvr_gsm),
                      dag=(i!=best_rlvr_gsm  and within_noise(rlvr_g_m, rlvr_g_s, rlvr_g_best_mean)))
        c4 = fmt_cell(rlvr_m_m, rlvr_m_s, bold=(i==best_rlvr_math),
                      dag=(i!=best_rlvr_math and within_noise(rlvr_m_m, rlvr_m_s, rlvr_m_best_mean)))
        c5 = fmt_cell(grpo_g_m, grpo_g_s, bold=(i==best_grpo_gsm),
                      dag=(i!=best_grpo_gsm  and within_noise(grpo_g_m, grpo_g_s, grpo_g_best_mean)))
        c6 = fmt_cell(grpo_m_m, grpo_m_s, bold=(i==best_grpo_math),
                      dag=(i!=best_grpo_math and within_noise(grpo_m_m, grpo_m_s, grpo_m_best_mean)))

        lines.append(f"    {label} & ${c1}$ & ${c2}$ & ${c3}$ & ${c4}$ & ${c5}$ & ${c6}$ \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 — Scaling: 4B vs 8B
# ---------------------------------------------------------------------------

def table2() -> str:
    """
    Scaling table: 4B RLVR / 8B RLVR / 8B GRPO for Retry Only and Reflect-Full.
    """
    conditions = [
        # (label,               4b_rlvr_pat,                       8b_rlvr_pat,                        8b_grpo_pat,                         mode)
        ("Retry Only",      "baseline_rlvr_retry-r8-seed{s}",    "baseline_rlvr_retry-8b-r8-seed{s}", None,                                "retry_only"),
        ("Reflect-Full",    "rrr-full-r8-seed{s}",               "rrr-full-8b-r8-seed{s}",            "rrr_grpo-full-8b-r8-seed{s}",        "reflect_full_retry"),
    ]

    # 4B GRPO for reference
    grpo_4b = {
        "Retry Only":   load_condition("baseline_grpo_retry-r8-seed{s}", "retry_only"),
        "Reflect-Full": load_condition("rrr_grpo-full-r8-seed{s}",       "reflect_full_retry"),
    }

    data = []
    for label, p4b, p8b_rlvr, p8b_grpo, mode in conditions:
        r4b      = load_condition(p4b,      mode)
        r8b_rlvr = load_condition(p8b_rlvr, mode)
        r8b_grpo = load_condition(p8b_grpo, mode) if p8b_grpo else {"gsm8k":(None,None),"math":(None,None)}
        data.append((label, r4b, grpo_4b[label], r8b_rlvr, r8b_grpo))

    lines = []
    lines.append(r"% Table 2: Scaling results (4B vs 8B, RLVR vs GRPO)")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Scaling results: 4B vs 8B model, RLVR vs GRPO.")
    lines.append(r"           Mean $\pm$ std across 3 seeds. \textbf{Bold} = row best.}")
    lines.append(r"  \label{tab:scaling}")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \begin{tabular}{lcccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{4}{c}{\textbf{4B (Qwen3-4B-Instruct-2507)}} & \multicolumn{4}{c}{\textbf{8B (Qwen3-8B)}} \\")
    lines.append(r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"    & \multicolumn{2}{c}{RLVR} & \multicolumn{2}{c}{GRPO} & \multicolumn{2}{c}{RLVR} & \multicolumn{2}{c}{GRPO} \\")
    lines.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"    \textbf{Condition} & GSM8K & MATH & GSM8K & MATH & GSM8K & MATH & GSM8K & MATH \\")
    lines.append(r"    \midrule")

    for label, r4b, r4b_grpo, r8b_rlvr, r8b_grpo in data:
        # Collect all 8 (mean,std) pairs; find row-best per dataset
        gsm_vals = [r4b["gsm8k"], r4b_grpo["gsm8k"], r8b_rlvr["gsm8k"], r8b_grpo["gsm8k"]]
        math_vals= [r4b["math"],  r4b_grpo["math"],  r8b_rlvr["math"],  r8b_grpo["math"]]

        best_gsm  = best_in_col(gsm_vals)
        best_math = best_in_col(math_vals)

        cells_gsm  = [fmt_cell(m, s, bold=(i==best_gsm))  for i,(m,s) in enumerate(gsm_vals)]
        cells_math = [fmt_cell(m, s, bold=(i==best_math)) for i,(m,s) in enumerate(math_vals)]

        row = f"    {label}"
        for cg, cm in zip(cells_gsm, cells_math):
            row += f" & ${cg}$ & ${cm}$"
        row += r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3 — Inference baselines: pass@k / majority@k
# ---------------------------------------------------------------------------

def table3() -> str:
    """
    Inference baselines: pass@k and majority@k for Baseline CoT at k=1,2,3,8,16.
    Shows RLVR-trained models as a comparison row.
    """
    # Load BoN data from compare_results scanning logic
    # We read the bon_*.jsonl files directly
    bon_dir = _RUNS
    ks      = [1, 2, 3, 8, 16]

    def load_bon_acc(k: int, metric: str, dataset: str) -> Optional[float]:
        path = bon_dir / f"bon_baseline_cot_n{k}_{dataset}.jsonl"
        if not path.exists():
            return None
        correct, total = 0, 0
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                total += 1
                if r.get(metric, False):
                    correct += 1
        return correct / total * 100 if total else None

    # RLVR Baseline CoT accuracy for reference (already computed)
    rlvr_cot = load_condition("baseline_rlvr_solve-r8-seed{s}", "baseline_solve")

    lines = []
    lines.append(r"% Table 3: Inference baselines (pass@k, majority@k) vs trained models")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Inference-only baselines (Baseline CoT, no training) vs RLVR-trained")
    lines.append(r"           model. pass@$k$ = oracle accuracy with $k$ samples;")
    lines.append(r"           maj@$k$ = majority-vote accuracy (deployable, no oracle).}")
    lines.append(r"  \label{tab:bon}")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \begin{tabular}{lcccccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{5}{c}{\textbf{GSM8K}} & \multicolumn{5}{c}{\textbf{MATH}} \\")
    lines.append(r"    \cmidrule(lr){2-6} \cmidrule(lr){7-11}")
    k_header = " & ".join([f"$k$={k}" for k in ks])
    lines.append(f"    \\textbf{{Model}} & {k_header} & {k_header} \\\\")
    lines.append(r"    \midrule")

    # pass@k rows
    gsm_pass, gsm_maj, math_pass, math_maj = [], [], [], []
    for k in ks:
        gp = load_bon_acc(k, "any_correct",      "gsm8k")
        gm = load_bon_acc(k, "majority_correct",  "gsm8k")
        mp = load_bon_acc(k, "any_correct",      "math")
        mm = load_bon_acc(k, "majority_correct",  "math")
        gsm_pass.append(gp); gsm_maj.append(gm)
        math_pass.append(mp); math_maj.append(mm)

    def fmt_v(v): return f"{v:.1f}" if v is not None else "---"

    gsm_pass_cells  = " & ".join([fmt_v(v) for v in gsm_pass])
    math_pass_cells = " & ".join([fmt_v(v) for v in math_pass])
    gsm_maj_cells   = " & ".join([fmt_v(v) for v in gsm_maj])
    math_maj_cells  = " & ".join([fmt_v(v) for v in math_maj])

    lines.append(f"    pass@$k$ (Baseline CoT) & {gsm_pass_cells} & {math_pass_cells} \\\\")
    lines.append(f"    maj@$k$  (Baseline CoT) & {gsm_maj_cells}  & {math_maj_cells}  \\\\")
    lines.append(r"    \midrule")

    # RLVR trained model single-sample accuracy
    rlvr_g_m, rlvr_g_s = rlvr_cot["gsm8k"]
    rlvr_m_m, rlvr_m_s = rlvr_cot["math"]
    rlvr_gsm_str  = fmt_cell(rlvr_g_m, rlvr_g_s) if rlvr_g_m else "---"
    rlvr_math_str = fmt_cell(rlvr_m_m, rlvr_m_s) if rlvr_m_m else "---"
    # Pad to 5 columns each
    gsm_rlvr_row  = f"\\textbf{{{rlvr_gsm_str}}} & --- & --- & --- & ---"
    math_rlvr_row = f"\\textbf{{{rlvr_math_str}}} & --- & --- & --- & ---"
    lines.append(f"    RLVR Baseline CoT ($k$=1) & {gsm_rlvr_row} & {math_rlvr_row} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bonus: Table 4 — GRPO capacity interaction (4B vs 8B)
# ---------------------------------------------------------------------------

def table4() -> str:
    """
    Compact 2×4 table: GRPO 4B vs 8B, RLVR 4B vs 8B — the capacity story.
    """
    rlvr_4b = load_condition("rrr-full-r8-seed{s}",         "reflect_full_retry")
    grpo_4b = load_condition("rrr_grpo-full-r8-seed{s}",    "reflect_full_retry")
    rlvr_8b = load_condition("rrr-full-8b-r8-seed{s}",      "reflect_full_retry")
    grpo_8b = load_condition("rrr_grpo-full-8b-r8-seed{s}", "reflect_full_retry")

    rows = [
        ("RLVR", rlvr_4b, rlvr_8b),
        ("GRPO", grpo_4b, grpo_8b),
    ]

    lines = []
    lines.append(r"% Table 4: Algorithm × model-size interaction (Reflect-Full+Retry)")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Algorithm $\times$ model-size interaction for Reflect-Full+Retry.")
    lines.append(r"           GRPO lags RLVR at 4B but recovers at 8B,")
    lines.append(r"           consistent with reward-sparsity explanation.}")
    lines.append(r"  \label{tab:capacity_interaction}")
    lines.append(r"  \setlength{\tabcolsep}{6pt}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{2}{c}{\textbf{4B}} & \multicolumn{2}{c}{\textbf{8B}} \\")
    lines.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r"    \textbf{Algorithm} & GSM8K & MATH & GSM8K & MATH \\")
    lines.append(r"    \midrule")

    # Find best per column
    gsm_vals  = [(r4b["gsm8k"], r8b["gsm8k"]) for _, r4b, r8b in rows]
    all_gsm4  = [r4b["gsm8k"] for _, r4b, _ in rows]
    all_gsm8  = [r8b["gsm8k"] for _, _, r8b in rows]
    all_math4 = [r4b["math"]  for _, r4b, _ in rows]
    all_math8 = [r8b["math"]  for _, _, r8b in rows]

    best_gsm4  = best_in_col(all_gsm4)
    best_gsm8  = best_in_col(all_gsm8)
    best_math4 = best_in_col(all_math4)
    best_math8 = best_in_col(all_math8)

    for i, (label, r4b, r8b) in enumerate(rows):
        c1 = fmt_cell(*r4b["gsm8k"], bold=(i==best_gsm4))
        c2 = fmt_cell(*r4b["math"],  bold=(i==best_math4))
        c3 = fmt_cell(*r8b["gsm8k"], bold=(i==best_gsm8))
        c4 = fmt_cell(*r8b["math"],  bold=(i==best_math8))
        lines.append(f"    {label} & ${c1}$ & ${c2}$ & ${c3}$ & ${c4}$ \\\\")

    # Delta row
    def delta(a_mean, b_mean):
        if a_mean is None or b_mean is None: return "---"
        d = b_mean - a_mean
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"

    d_gsm4  = delta(rows[1][1]["gsm8k"][0], rows[0][1]["gsm8k"][0])
    d_math4 = delta(rows[1][1]["math"][0],  rows[0][1]["math"][0])
    d_gsm8  = delta(rows[1][2]["gsm8k"][0], rows[0][2]["gsm8k"][0])
    d_math8 = delta(rows[1][2]["math"][0],  rows[0][2]["math"][0])
    lines.append(r"    \midrule")
    lines.append(f"    $\\Delta$ (RLVR$-$GRPO) & ${d_gsm4}$ & ${d_math4}$ & ${d_gsm8}$ & ${d_math8}$ \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--table", type=int, choices=[1, 2, 3, 4], default=None,
                    help="Which table to generate (default: all).")
    ap.add_argument("--out", default=None,
                    help="Output .tex file (default: stdout).")
    args = ap.parse_args()

    generators = {1: table1, 2: table2, 3: table3, 4: table4}
    tables_to_run = [args.table] if args.table else [1, 2, 3, 4]

    parts = []
    for t in tables_to_run:
        parts.append(f"% {'='*70}")
        parts.append(f"% TABLE {t}")
        parts.append(f"% {'='*70}")
        parts.append(generators[t]())
        parts.append("")

    output = "\n".join(parts)

    if args.out:
        out_path = _REPO_ROOT / args.out
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote {len(parts)} lines to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
