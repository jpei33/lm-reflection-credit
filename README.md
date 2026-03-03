# LM Reflection + Step-Local Credit Assignment

This repo investigates whether **step-local credit assignment** (predicting where reasoning fails) can outperform **outcome-gated reflection** (RRR-style) on math reasoning with curriculum learning.

## Methods
- **Baseline RLVR-CoT**: outcome reward on final answer
- **RRR (Outcome-gated reflection + retry)**: reward reflection tokens only when retry succeeds
- **Step-local credit assignment**: learn/predict mistake step index and weight policy gradients by step region

## Repo layout
- `src/rrr/` — reflection + retry training and prompting
- `src/step_credit/` — mistake-step labeler + locator + weighted RL
- `src/eval/` — first-try and retry evaluation
- `configs/` — experiment configs
- `scripts/` — entrypoints for training/eval
- `notebooks/` — plotting and analysis

## Quickstart
1. Create venv and install deps
2. Prepare data
3. Run baseline, RRR, and step-credit experiments
4. Generate plots

## Phase 0: Pretrain Inference Results (Qwen/Qwen2.5-0.5B-Instruct)

All runs logged to `results/` (ignored by git). Use `notebooks/` to plot.

We evaluate four inference strategies — **baseline**, **retry-only**, **reflection-full**, **reflection-plan**, and **reflection-tail** — on two reasoning benchmarks:

- GSM8K-200 (grade-school arithmetic)
- MATH-200 (competition math)

All experiments use identical prompts, decoding limits, and model weights. Differences arise only from inference strategy.

---

### GSM8K-200

| Method | Strict Accuracy | Δ vs Baseline | Tokens / Example | Latency / Example |
|--------|-----------------|---------------|-------------------|-------------------|
| Baseline | 0.085 | — | **390** | **7.23 s** |
| Retry | 0.165 | +0.080 | 727 | 13.14 s |
| Reflect-Full | 0.115 | +0.030 | 1111 | 16.72 s |
| Reflect-Plan | **0.185** | **+0.100** | 893 | 17.27 s |
| Reflect-Tail | 0.175 | +0.090 | 948 | 17.30 s |

**Takeaways**

- Structured reflection improves reasoning.
- Planning-style reflection works best.
- Long reflection hurts efficiency.
- Retry alone already gives large gains.

---

### MATH-200

| Method | Strict Accuracy | Δ vs Baseline | Tokens / Example | Latency / Example |
|--------|-----------------|---------------|-------------------|-------------------|
| Baseline | 0.150 | — | **632** | **14.24 s** |
| Retry | 0.255 | +0.105 | 1223 | 27.51 s |
| Reflect-Full | 0.255 | +0.105 | 1881 | 28.02 s |
| Reflect-Tail | **0.265** | **+0.115** | 1509 | 28.08 s |
| Reflect-Plan | 0.240 | +0.090 | 1433 | 27.95 s |

**Takeaways**

- Hard math benefits more from search than reflection.
- Tail reflection is best for difficult reasoning.
- Full reflection adds cost but little gain.

---

### Plots

<p align="center">
  <img src="figures/acc_strict_by_method.png" width="48%" />
  <img src="figures/tokens_by_method.png" width="48%" />
</p>

<p align="center">
  <img src="figures/latency_by_method.png" width="48%" />
  <img src="figures/tradeoff_acc_vs_tokens.png" width="48%" />
</p>


---

### Cross-Dataset Insights

- Retry is a strong baseline.
- Reflection must be constrained to help.
- Short reflections outperform long ones.
- Best method depends on task difficulty.

| Dataset | Best Method |
|--------|-------------|
| GSM8K | Reflect-Plan |
| MATH | Reflect-Tail |

---

### Key Finding

> LLM reasoning performance appears largely search-limited rather than capability-limited.

Reflection mainly helps by guiding search rather than generating new reasoning ability.

---

## Phase 1: SFT + RLVR Fine-tuning (Qwen3-4B-Instruct)

We train a LoRA adapter on top of `Qwen/Qwen3-4B-Instruct-2507` using supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR). RLVR uses rejection-sampling fine-tuning (RFT): at each step, the model samples outputs and trains only on correct ones with a binary reward signal from a math verifier.

Each training run is 500 steps on a mixed GSM8K + MATH training set. Evaluation uses 200 held-out examples per dataset.

All conditions are evaluated with a **single forward pass** (no retry at inference), so token costs are identical across conditions and accuracy differences reflect purely what each training method taught the model.

### GSM8K-200 (single-pass eval)

| Condition | SFT | RLVR | Δ (RLVR) |
|---|---|---|---|
| Baseline CoT | 34.5% | 31.5% | -3.0% |
| Retry Only | 32.0% | **34.5%** | **+2.5%** |
| Reflect-Full | 28.5% | 30.5% | +2.0% |
| Reflect-Plan | 27.0% | 24.5% | -2.5% |
| Step Credit | — | **32.5%** | — |

### MATH-200 (single-pass eval)

| Condition | SFT | RLVR | Δ (RLVR) |
|---|---|---|---|
| Baseline CoT | 15.5% | 16.5% | +1.0% |
| Retry Only | **20.5%** | **23.0%** | **+2.5%** |
| Reflect-Full | 16.5% | 18.0% | +1.5% |
| Reflect-Plan | 16.5% | 15.5% | -1.0% |
| Step Credit | — | 17.0% | — |

### MATH-200 by Difficulty Level (RLVR, single-pass)

| Condition | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Baseline CoT | 38.1% | 30.0% | 19.2% | 12.2% | 1.8% |
| Retry Only | **52.4%** | **36.7%** | **25.0%** | 12.2% | **10.7%** |
| Reflect-Full | 47.6% | 30.0% | 21.2% | 9.8% | 3.6% |
| Reflect-Plan | 42.9% | 30.0% | 19.2% | 4.9% | 1.8% |
| Step Credit | 38.1% | 30.0% | 19.2% | **14.6%** | 1.8% |

---

### GSM8K-200 by Difficulty (RLVR, single-pass)

GSM8K has no native difficulty labels. Difficulty is proxied by the number of non-trivial computation steps in the reference solution (i.e. `<<a op b=c>>` annotations where `a ≠ c`).

| Condition | Easy (≤2 steps, n=73) | Medium (3–4 steps, n=90) | Hard (5+ steps, n=37) |
|---|---|---|---|
| Baseline CoT | **57.5%** | 21.1% | 5.4% |
| Retry Only | **60.3%** | **24.4%** | **8.1%** |
| Reflect-Full | 53.4% | 22.2% | 5.4% |
| Reflect-Plan | 50.7% | 12.2% | 2.7% |
| Step Credit | 56.2% | 23.3% | **8.1%** |

<p align="center">
  <img src="figures/gsm8k_difficulty_stratified.png" width="92%" />
</p>

**Takeaways from difficulty stratification:**

- Retry Only RLVR leads on Easy and Medium, but **Step Credit ties Retry Only on Hard (8.1%)**, both outperforming all other conditions on the hardest GSM8K problems. This is consistent with the step-local credit hypothesis.
- Reflect-Plan collapses on Medium (12.2%), far behind all other conditions, suggesting planning-style prompts are actively harmful with RLVR at this scale.
- The Hard tier (n=37) is small — 8.1% is 3 correct examples. Multi-seed runs are needed to confirm.
- The MATH L4 result is the strongest signal: Step Credit (14.6%) beats all other conditions at Level 4, the tier most accessible to a 4B model while still being genuinely hard.

---

### RLVR Training Dynamics

The plot below shows rollout hit rate (fraction of training rollouts yielding a correct answer) smoothed over a 40-step window. Stars (★) mark the final held-out eval accuracy at step 500 for each condition.

<p align="center">
  <img src="figures/rlvr_training_curves.png" width="95%" />
</p>

**Key observations:**

- **Reflect-Plan (MATH) collapses around step 200**: the model stops producing correct retry trajectories entirely, explaining the −1.0% eval regression. The training signal dies before the run ends.
- **Retry Only and Step Credit converge to the highest MATH hit rates** by step 500, consistent with their leading eval numbers.
- **GSM8K conditions stay tightly clustered** (~40–60%), with no condition clearly pulling away — the task is simply easier and all methods find correct solutions at similar rates.
- **Training hit rate ≫ eval accuracy** across the board (expected): the model sees easier in-distribution problems during training rollouts vs. the held-out eval set.

---

### Efficiency-Accuracy Tradeoff (Phase 1)

All Phase 1 conditions are trained for 500 steps and evaluated with a **single forward pass**, so inference token cost is essentially identical across conditions (~5–6 completion tokens on GSM8K, ~8 on MATH). The plot below shows that accuracy differences are driven entirely by training method, not inference compute.

<p align="center">
  <img src="figures/phase2_efficiency_accuracy.png" width="90%" />
</p>

Token counts annotated above each RLVR bar confirm near-identical inference cost across all conditions.

---

### Takeaways

- **Retry Only RLVR is the most consistently positive condition**: +2.5% on both GSM8K and MATH, best single-pass accuracy on both datasets. Simple retry-based RL generalizes well to first-pass reasoning.
- **MATH responds to RLVR at 500 steps** where it did not at 200: Retry Only (+2.5%), Reflect-Full (+1.5%), and Baseline CoT (+1.0%) all improve, suggesting MATH requires more training signal to move.
- **Reflect-Plan degrades everywhere**: −2.5% on GSM8K, −1.0% on MATH, worst condition at every hard MATH level. Planning-style prompts appear incompatible with RFT at this scale.
- **Baseline CoT regresses on GSM8K (−3.0%)**: RFT without structured output supervision causes drift, consistent with the hypothesis that outcome-only reward on single-pass solves is too sparse.
- **Step Credit ties Retry Only on Hard GSM8K (8.1%)** and leads all conditions on MATH L4 (14.6%), supporting the prediction that step-local credit assignment is more effective on harder problems where the error region is meaningful.
- **Step Credit (32.5% GSM8K) holds competitive without an SFT warm-start on retry data**, matching or beating most RLVR conditions despite starting from a baseline SFT checkpoint.
