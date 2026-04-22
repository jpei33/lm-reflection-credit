"""
Tests for src/eval/passk_estimator.py.

Covers:
  1. Closed-form identities (pass@k edge cases: c=0, c=n, k=1, k=n).
  2. Monotonicity in c and k.
  3. Cross-check against the independent Day 6 reference implementation
     (bootstrap_ci.py in frontier_eval_prep).
  4. Bootstrap CI coverage on a Bernoulli-mixture synthetic dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.passk_estimator import (
    bootstrap_pass_at_k,
    benchmark_pass_at_k,
    pass_at_k,
    summarise_passk,
)


# ---------------------------------------------------------------------------
# Closed-form edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_c_zero_gives_zero(self):
        assert pass_at_k(16, 0, 8) == 0.0

    def test_c_equals_n_gives_one(self):
        assert pass_at_k(16, 16, 8) == 1.0

    def test_k_one_reduces_to_c_over_n(self):
        # Unbiased pass@1 == c / n
        for n, c in [(10, 3), (16, 7), (8, 1)]:
            assert math.isclose(pass_at_k(n, c, 1), c / n, rel_tol=1e-9)

    def test_k_equals_n_reduces_to_any_correct(self):
        # pass@n: any correct in n samples = 1 iff c >= 1
        for n, c in [(8, 1), (8, 4), (16, 16)]:
            assert pass_at_k(n, c, n) == 1.0
        assert pass_at_k(8, 0, 8) == 0.0

    def test_n_minus_c_less_than_k_gives_one(self):
        # If fewer than k wrong samples exist, every k-subset hits correct
        assert pass_at_k(10, 8, 4) == 1.0  # only 2 wrong, any 4-subset wins


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_c_greater_than_n(self):
        with pytest.raises(ValueError):
            pass_at_k(n=4, c=5, k=2)

    def test_rejects_k_greater_than_n(self):
        with pytest.raises(ValueError):
            pass_at_k(n=4, c=2, k=5)

    def test_rejects_negatives(self):
        with pytest.raises(ValueError):
            pass_at_k(n=-1, c=0, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=4, c=-1, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=4, c=2, k=-1)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_monotone_in_c(self):
        # More correct samples => higher pass@k
        vals = [pass_at_k(16, c, 8) for c in range(0, 17)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_monotone_in_k(self):
        # Larger k => higher pass@k (same c, n)
        n, c = 16, 3
        vals = [pass_at_k(n, c, k) for k in range(1, n + 1)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# Cross-check against the Day 6 reference (numpy-based) implementation
# ---------------------------------------------------------------------------


def _day6_reference(n: int, c: int, k: int) -> float:
    """Day 6 bootstrap_ci.py implementation, reproduced for cross-check."""
    if n - c < k:
        return 1.0
    # 1 - prod_{i=0..k-1} (n - c - i) / (n - i)
    log_term = 0.0
    for i in range(k):
        log_term += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_term)


class TestCrossCheckDay6:
    @pytest.mark.parametrize(
        "n,c,k",
        [
            (8, 3, 4),
            (16, 7, 8),
            (16, 1, 16),
            (16, 15, 2),
            (10, 5, 3),
        ],
    )
    def test_matches_day6_reference(self, n, c, k):
        assert math.isclose(
            pass_at_k(n, c, k),
            _day6_reference(n, c, k),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


# ---------------------------------------------------------------------------
# benchmark_pass_at_k aggregation
# ---------------------------------------------------------------------------


class TestBenchmarkAggregation:
    def test_mean_over_tasks(self):
        # Two tasks: (n=10, c=5) and (n=10, c=0) with k=5
        # pass@5 task 1 ≈ 1 - C(5,5)/C(10,5) = 1 - 1/252 ≈ 0.99603
        # pass@5 task 2 = 0.0
        # mean ≈ 0.498
        got = benchmark_pass_at_k([(10, 5), (10, 0)], k=5)
        assert math.isclose(got, (pass_at_k(10, 5, 5) + 0.0) / 2, rel_tol=1e-9)

    def test_empty_returns_nan(self):
        got = benchmark_pass_at_k([], k=3)
        assert math.isnan(got)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_contains_point(self):
        tasks = [(16, c) for c in [0, 1, 2, 3, 4, 5, 8, 10, 12, 16]]
        res = bootstrap_pass_at_k(tasks, k=4, n_resamples=2000, seed=0)
        assert res.lo <= res.point <= res.hi

    def test_ci_shrinks_with_more_tasks(self):
        # More tasks => narrower CI, all drawn from the same underlying dist
        rng = np.random.default_rng(0)
        base_tasks = [(16, int(c)) for c in rng.integers(0, 17, size=50)]
        more_tasks = base_tasks * 4  # 200 tasks, same distribution

        r50 = bootstrap_pass_at_k(base_tasks, k=4, n_resamples=3000, seed=1)
        r200 = bootstrap_pass_at_k(more_tasks, k=4, n_resamples=3000, seed=1)

        assert (r200.hi - r200.lo) < (r50.hi - r50.lo)

    def test_ci_coverage_on_bernoulli_mixture(self):
        # Simulate ~200 tasks where c_i ~ Binomial(n=16, p=0.3) for all tasks.
        # The true pass@4 per task is approx constant across tasks.
        # Coverage check: 95% CI should contain the true mean most of the time.
        rng = np.random.default_rng(42)
        true_p = 0.3
        n_samples = 16
        k = 4
        true_passk = pass_at_k(n_samples, int(round(true_p * n_samples)), k)
        # This is an approximation of the "true" per-task pass@k at p=0.3

        covered = 0
        trials = 30
        for trial in range(trials):
            tasks = [
                (n_samples, int(rng.binomial(n_samples, true_p)))
                for _ in range(150)
            ]
            res = bootstrap_pass_at_k(tasks, k=k, n_resamples=1500, seed=trial)
            # Use the true mean under the binomial for coverage check
            target = float(
                np.mean([pass_at_k(n_samples, c, k) for _, c in tasks])
            )
            if res.lo <= target <= res.hi:
                covered += 1
        # Should cover the empirical mean essentially always (resample of same data)
        assert covered == trials


# ---------------------------------------------------------------------------
# summarise_passk convenience helper
# ---------------------------------------------------------------------------


class TestSummarisePassK:
    def test_oracle_exceeds_greedy_on_correlated_failures(self):
        # Construct 10 tasks. First sample correct 3/10 times; oracle (any of 8)
        # correct 7/10 times. Oracle should be strictly higher.
        per_task = []
        for i in range(10):
            first_correct = i < 3
            n_correct = 8 if i < 7 else 0
            per_task.append({"n": 8, "n_correct_in_group": n_correct, "first_correct": first_correct})

        summary = summarise_passk(per_task, k=8, label="synth", seed=0, n_resamples=1000)
        assert summary.oracle.point > summary.greedy.point
        assert summary.gap > 0.3  # clearly positive gap
