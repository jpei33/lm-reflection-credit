# Phase 15 — pass@k Oracle Training Sweep (Day 22 protocol)

**Status:** protocol specified, sampling pending a GPU/Tinker run.
**Driver:**   `scripts/run_passk_training_sweep.py`
**Plotter:**  `scripts/plot_passk_training_sweep.py`
**Estimator:** `src/eval/passk_estimator.py` (tested in `tests/test_passk_estimator.py`)

---

## 1. Motivation

Phase 11 (`EXPERIMENTS.md`) measured pass@k / majority@k at a **single** checkpoint per condition. That rules out the compute-equivalence alternative explanation for RLVR lift, but does not tell us **what changes during training**. Specifically: does the oracle ceiling (any-of-k correct) rise, does the greedy floor (pass@1) rise, or both — and at what relative rate?

The oracle-vs-greedy gap is a direct diagnostic for the elicitation-vs-capability decomposition of RLVR gains:

| Pattern over training | Interpretation |
|---|---|
| gap closes (oracle flat, greedy rises) | RLVR is **elicitation**: the capability was already reachable with search; training is teaching the model to reach it on the first try. Implies correlated failures pre-training, independent successes post-training. |
| gap holds (oracle rises, greedy rises equally) | RLVR is **capability**: the ceiling genuinely lifts. More inference tokens won't substitute for training. |
| gap widens (oracle rises faster than greedy, or greedy collapses) | RLVR is adding capability at the cost of elicitation — e.g. mode collapse or overconfident single-sample behaviour. Red flag, worth investigating. |

This is the experiment Day 22 of the OpenAI Frontier Evals prep study plan calls for. It is directly nameable in application materials and interviews as concrete experimental evidence about eval methodology (pass@k as a diagnostic for elicitation vs. capability).

## 2. Design

### 2.1 Sampling protocol (per cell)

| Setting | Value | Rationale |
|---|---|---|
| k (samples per problem) | **16** | Matches Phase 11's upper bound. Produces tight CIs per task given n=16 binary outcomes. |
| sampling | temperature = 0.7, top_p = 0.95 | Same as Phase 11 — preserves comparability. |
| max_new_tokens | 512 | Same as existing `eval_best_of_n.py` default. |
| greedy proxy | sample\_idx = 0 of the BoN group | Avoids a separate T=0 decode; uses the same distribution as pass@k so the gap is apples-to-apples. Caveat: this is "T=0.7 pass@1", not a true greedy decode. |
| datasets | gsm8k_test_200, math_test_200 | Reused from existing eval infrastructure. |
| seeds | 42 (first pass) | Single seed keeps compute cheap; subsequent seeds can be added if the gap is ambiguous. |

### 2.2 Cells — 4B sweep (6 cells × 2 datasets)

| Condition | Step label | Checkpoint | Exists on disk |
|---|---|---|---|
| Reflect-Full (RLVR) | SFT (step 0) | `qwen3-4b-reflect_full_retry-r8-seed42` | ✓ |
| Reflect-Full (RLVR) | step 350 | `rrr-grounded-r8-seed42-v7-step350` | ✓ |
| Reflect-Full (RLVR) | step 500 | `rrr-grounded-r8-seed42-v7` | ✓ |
| Baseline CoT (RLVR) | SFT (step 0) | `qwen3-4b-baseline_solve-r8-seed42` | ✓ |
| Baseline CoT (RLVR) | step 300 | `baseline_rlvr_solve-r8-seed42-step300` | ✓ |
| Baseline CoT (RLVR) | step 500 | `baseline_rlvr_solve-r8-seed42` | ✓ |

All six checkpoints exist on disk. Mid-training steps differ slightly (300 vs 350) because those are the closest checkpoints available for each condition; both are ~60–70% through a 500-step training run.

### 2.3 Estimator

Per task, we observe (n = 16, c = n_correct_in_group, first_correct). For each cell we compute:

- **oracle pass@16** — unbiased estimator (Chen et al. 2021), averaged over tasks
- **greedy pass@1** — `mean(first_correct)` across tasks (algebraically equal to unbiased pass@1 on the first sample)
- **gap** = oracle − greedy
- **95% bootstrap CI** — resample tasks with replacement 10 000 times (task-level variance only; sampler variance absorbed into the per-task binomial).

Formulas and the log-space implementation live in `src/eval/passk_estimator.py`. The unbiased estimator and its equivalence to the Day 6 reference are checked in `tests/test_passk_estimator.py`.

### 2.4 Compute budget

One cell ≈ 200 problems × 16 samples × 512 max tokens. With Tinker's parallel `sample()` fan-out this is the same cost per cell that Phase 11 already paid at N=16. Six cells × 2 datasets = 12 BoN files total. Half (N=16 on seed42 for Baseline SFT and Reflect-Full final) is already on disk from Phase 11; the sweep script will **resume** into those files so no work is duplicated.

**Incremental work for Phase 15** = 10 cells × 200 problems × 16 samples ≈ 32 k sample calls. On Tinker this is in the low-hour range, not days.

## 3. How to run

```bash
# 1. Sanity: check all 6 checkpoint sidecars exist (dry-run)
python scripts/run_passk_training_sweep.py --model 4b --dry-run

# 2. Run the full sweep (resume picks up existing Phase 11 cells for free)
python scripts/run_passk_training_sweep.py --model 4b --k 16 --seed 42 --resume

# 3. Regenerate the plot + summary JSON from the BoN files
python scripts/plot_passk_training_sweep.py

# 3b. Smoke-test the plot pipeline with synthetic data (no sampling cost)
python scripts/plot_passk_training_sweep.py --synthetic
```

Outputs:
- `results/runs/<run_name>_bon_n16_{gsm8k,math}.jsonl` — per-cell raw data
- `figures/fig2b_passk_training_sweep.png` — the 2×2 panel figure
- `results/analysis/passk_training_sweep_summary.json` — machine-readable summary

## 4. Decision points to resolve before running

1. **Seeds.** Is a single seed (42) sufficient, or should we run seeds 0 and 1 too for the final-step cells (to get error bars matching Fig. 1)?
2. **8B extension.** Is the 4B result worth running on 8B? The 8B Reflect-Full checkpoints exist at step {50, 150, 250, 350, 500} for seed0; the sweep driver has a `--model 8b` path that's staged but not enabled by default.
3. **Greedy definition.** Stick with `sample_idx=0 at T=0.7`, or run a separate T=0 decode per checkpoint for a true greedy measurement? The current choice is cheaper and keeps the metric family consistent; the alternative is cleaner but doubles eval cost.

## 5. Expected results (from prior context)

From Phase 11 and the reflect_full_retry evaluation:

- At step 0 (SFT): oracle pass@16 ≈ 51% (GSM8K) / 37% (MATH); greedy ≈ 24% / 16%. **Gap ≈ 27 pp / 21 pp.**
- At step 500 Reflect-Full: greedy pass@1 ≈ 32.5% (GSM8K) / 18.5% (MATH). Oracle not yet measured — hypothesis: stays roughly flat (~50% / ~36%), so **gap shrinks to ≈ 18 pp / 18 pp**.
- At step 500 Baseline CoT RLVR: greedy pass@1 ≈ 31.5% (GSM8K) / 16.5% (MATH). Same hypothesis: oracle roughly flat, **gap shrinks to ≈ 19 pp / 19 pp**.

If the hypothesis holds, both training trajectories look like *elicitation*, not capability expansion. This is consistent with the project's existing thesis: "failure modes are correlated → reflection is an elicitation problem, not a search problem." Confirming this with a training-step sweep is the leap from a cross-sectional claim to a longitudinal one.

## 6. Connection to OpenAI application

The interview framing: pass@k is not a static benchmark number — it is a **diagnostic** that, read longitudinally, separates what a training signal is contributing to a model. Every eval team eventually needs this decomposition, because pass@1 (the shipping metric) hides whether residual headroom is reachable with better training or with better sampling. Phase 15 is the minimal experiment that demonstrates the distinction in practice.

See `paper_draft_v3.md` / `paper_full_draft.md` for how this fits the lm-reflection-credit narrative.
