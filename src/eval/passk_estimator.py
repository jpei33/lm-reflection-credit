"""
passk_estimator.py
------------------
Unbiased pass@k estimator (Chen et al., Codex paper, 2021) plus a bootstrap CI
helper. Pure functions — no I/O, no side effects. Used by the pass@k oracle
training sweep (scripts/plot_passk_training_sweep.py) and the existing
compare_results pipeline.

Why the unbiased estimator instead of a plain mean?
------------------------------------------------------
Given n samples per problem and c of them correct, the probability that at
least one of k sampled-without-replacement samples is correct is

    pass@k = 1 - C(n - c, k) / C(n, k)

This is the exact expectation under uniform sampling of k from the n drawn
samples. A plain "any correct in the first k" estimate is biased when c is
small (because it conditions on sample ordering). Log-space arithmetic keeps
C(n - c, k) / C(n, k) numerically stable for n = 16 and beyond.

The bootstrap CI resamples the N tasks (with replacement) 10k times by
default, computing pass@k on each resample. This captures task-level
variance — the dominant source of uncertainty — but not sampler variance. If
you need sampler variance too, re-sample (n, c) pairs from per-task binomial
distributions; that is usually overkill.

References:
  - Chen et al. 2021, "Evaluating LLMs Trained on Code" (Codex paper)
  - frontier_eval_prep/Code/bootstrap_ci.py (Justin's Day 6 reference impl)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Unbiased pass@k estimator
# ---------------------------------------------------------------------------


def _log_comb(n: int, k: int) -> float:
    """log(C(n, k)) via lgamma. Returns -inf if k > n or k < 0."""
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k from Chen et al. (2021).

    Parameters
    ----------
    n : total samples drawn for the problem (n >= k)
    c : number of those samples that were correct (0 <= c <= n)
    k : k for pass@k

    Returns
    -------
    Estimated probability that a random subset of k out of n samples contains
    at least one correct sample. In [0, 1].
    """
    if n < 0 or c < 0 or k < 0:
        raise ValueError(f"pass_at_k requires non-negative n, c, k; got {n=}, {c=}, {k=}")
    if c > n:
        raise ValueError(f"pass_at_k requires c <= n; got c={c}, n={n}")
    if k > n:
        raise ValueError(f"pass_at_k requires k <= n; got k={k}, n={n}")

    if c == 0:
        return 0.0
    if n - c < k:
        # Every subset of size k must contain at least one correct sample.
        return 1.0

    # 1 - C(n - c, k) / C(n, k)  in log-space
    log_ratio = _log_comb(n - c, k) - _log_comb(n, k)
    return float(1.0 - np.exp(log_ratio))


def benchmark_pass_at_k(task_results: Sequence[tuple[int, int]], k: int) -> float:
    """
    Aggregate pass@k across a set of tasks.

    Parameters
    ----------
    task_results : iterable of (n_i, c_i) per task
    k : k for pass@k

    Returns
    -------
    Mean over tasks of pass_at_k(n_i, c_i, k).
    """
    vals = [pass_at_k(n, c, k) for n, c in task_results]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Bootstrap CI over tasks
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    point: float  # pass@k on the full task set
    lo: float     # lower percentile bound
    hi: float     # upper percentile bound
    n_tasks: int
    n_resamples: int
    k: int
    alpha: float  # total two-sided mass outside [lo, hi]

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n_tasks": self.n_tasks,
            "n_resamples": self.n_resamples,
            "k": self.k,
            "alpha": self.alpha,
        }


def bootstrap_pass_at_k(
    task_results: Sequence[tuple[int, int]],
    k: int,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> BootstrapResult:
    """
    Bootstrap CI for pass@k over tasks. Resamples N tasks with replacement.

    Parameters
    ----------
    task_results : per-task (n_i, c_i) pairs
    k            : k for pass@k
    n_resamples  : bootstrap draws (default 10k)
    alpha        : 1 - coverage, e.g. 0.05 for 95% CI
    seed         : RNG seed for reproducibility (None = nondeterministic)

    Returns
    -------
    BootstrapResult with point estimate and percentile CI.
    """
    task_results = list(task_results)
    n = len(task_results)
    if n == 0:
        raise ValueError("bootstrap_pass_at_k needs at least one task")

    per_task = np.array([pass_at_k(nn, cc, k) for nn, cc in task_results], dtype=np.float64)
    point = float(per_task.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = per_task[idx].mean(axis=1)

    lo = float(np.quantile(resample_means, alpha / 2))
    hi = float(np.quantile(resample_means, 1 - alpha / 2))

    return BootstrapResult(
        point=point,
        lo=lo,
        hi=hi,
        n_tasks=n,
        n_resamples=n_resamples,
        k=k,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Convenience: summarise oracle (pass@k) and greedy (pass@1) jointly
# ---------------------------------------------------------------------------


@dataclass
class PasskSummary:
    """Joint oracle / greedy / gap for a single (condition, step, dataset) cell."""
    oracle: BootstrapResult     # pass@k across all k samples
    greedy: BootstrapResult     # pass@1 using only the first sample (greedy proxy)
    gap: float                   # oracle.point - greedy.point
    label: str                   # e.g. "Reflect-Full, step 500, GSM8K"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "oracle": self.oracle.as_dict(),
            "greedy": self.greedy.as_dict(),
            "gap": self.gap,
        }


def summarise_passk(
    per_task: Sequence[dict],
    k: int,
    *,
    label: str,
    seed: int | None = 0,
    n_resamples: int = 10_000,
) -> PasskSummary:
    """
    Given a list of per-task dicts with fields {"n_correct_in_group": c, "n": n,
    "first_correct": bool}, compute oracle pass@k and greedy pass@1 with CIs.

    The "greedy" proxy is the correctness of the first sample in the BoN group
    (sample_idx=0). This is not a true temperature=0 greedy decode — it is
    pass@1 at T=0.7 — but it uses the same sampling distribution as pass@k so
    the oracle-vs-greedy gap is apples-to-apples.
    """
    oracle_inputs = [(int(r["n"]), int(r["n_correct_in_group"])) for r in per_task]
    greedy_inputs = [(1, int(bool(r["first_correct"]))) for r in per_task]

    oracle = bootstrap_pass_at_k(oracle_inputs, k=k, seed=seed, n_resamples=n_resamples)
    greedy = bootstrap_pass_at_k(greedy_inputs, k=1, seed=seed, n_resamples=n_resamples)

    return PasskSummary(
        oracle=oracle,
        greedy=greedy,
        gap=oracle.point - greedy.point,
        label=label,
    )


__all__ = [
    "pass_at_k",
    "benchmark_pass_at_k",
    "bootstrap_pass_at_k",
    "summarise_passk",
    "BootstrapResult",
    "PasskSummary",
]
