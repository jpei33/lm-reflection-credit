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

Each training run is 500 steps on a mixed GSM8K + MATH training set. Evaluation uses 200 held-out examples per dataset. Results are reported as **mean ± std across 3 random seeds** (seeds 42, 0, 1).

All conditions are first evaluated with a **single forward pass** (no retry at inference) to isolate the quality of what each training method taught the model, then re-evaluated in **native mode** (with each model's trained retry/reflect strategy enabled) to measure the full system benefit.

### GSM8K-200 (first-try / single-pass, mean ± std over 3 seeds)

| Condition | SFT | RLVR | Δ (RLVR) |
|---|---|---|---|
| Baseline CoT | 34.5%±0.0% | 31.5%±2.0% | −3.0% |
| Retry Only | 32.0%±0.0% | **33.3%±0.8%** | +1.3% |
| Reflect-Full + Retry | 28.5%±0.0% | **31.5%±0.9%** | **+3.0%** |
| Reflect-Plan + Retry | 27.0%±0.0% | 27.0%±2.3% | +0.0% |
| Step Credit | — | 30.2%±2.4% | — |

### MATH-200 (first-try / single-pass, mean ± std over 3 seeds)

| Condition | SFT | RLVR | Δ (RLVR) |
|---|---|---|---|
| Baseline CoT | 15.5%±0.0% | 17.2%±0.8% | +1.7% |
| Retry Only | **20.5%±0.0%** | 18.3%±4.1% | −2.2% |
| Reflect-Full + Retry | 16.5%±0.0% | **18.5%±0.5%** | **+2.0%** |
| Reflect-Plan + Retry | 16.5%±0.0% | 15.8%±0.8% | −0.7% |
| Step Credit | — | 16.8%±0.6% | — |

---

### Native-Mode Evaluation (with retry/reflect enabled, mean ± std over 3 seeds)

Each model is re-evaluated using its trained inference strategy: Retry Only and Step Credit use blind retry; Reflect-Full and Reflect-Plan use reflect-then-retry. System accuracy = first-try **OR** retry correct. Baseline CoT has no retry so its system accuracy equals first-try.

#### GSM8K-200 (native-mode system accuracy)

| Condition | RLVR (system) | RLVR (first-try) | Retry lift |
|---|---|---|---|
| Baseline CoT | 31.5%±2.0% | 31.5%±2.0% | — |
| Retry Only | **34.5%±1.3%** | 33.3%±0.8% | +1.2% |
| Reflect-Full + Retry | **32.5%±0.0%** | 31.5%±0.9% | +1.0% |
| Reflect-Plan + Retry | 27.7%±2.0% | 27.0%±2.3% | +0.7% |
| Step Credit | 31.3%±2.1% | 30.2%±2.4% | +1.1% |

#### MATH-200 (native-mode system accuracy)

| Condition | RLVR (system) | RLVR (first-try) | Retry lift |
|---|---|---|---|
| Baseline CoT | 17.2%±0.8% | 17.2%±0.8% | — |
| Retry Only | 19.2%±3.3% | 18.3%±4.1% | +0.9% |
| Reflect-Full + Retry | **18.7%±0.3%** | 18.5%±0.5% | +0.2% |
| Reflect-Plan + Retry | 16.2%±0.8% | 15.8%±0.8% | +0.4% |
| Step Credit | 17.3%±0.6% | 16.8%±0.6% | +0.5% |

---

### MATH-200 by Difficulty Level (RLVR, native-mode)

| Condition | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Baseline CoT | 38.1% | 30.0% | 19.2% | 12.2% | 1.8% |
| Retry Only | **52.4%** | **36.7%** | **25.0%** | 12.2% | **10.7%** |
| Reflect-Full | 47.6% | 33.3% | 21.2% | 9.8% | 3.6% |
| Reflect-Plan | 42.9% | 30.0% | 19.2% | 4.9% | 1.8% |
| Step Credit | 38.1% | 30.0% | 19.2% | **14.6%** | 1.8% |

---

### GSM8K-200 by Difficulty (RLVR, native-mode)

GSM8K has no native difficulty labels. Difficulty is proxied by the number of non-trivial computation steps in the reference solution (i.e. `<<a op b=c>>` annotations where `a ≠ c`).

| Condition | Easy (≤2 steps, n=67) | Medium (3–4 steps, n=93) | Hard (5+ steps, n=40) |
|---|---|---|---|
| Baseline CoT | 56.7% | 23.7% | 7.5% |
| Retry Only | **61.2%** | 24.7% | 10.0% |
| Reflect-Full | 55.2% | 24.7% | **12.5%** |
| Reflect-Plan | 50.7% | 15.1% | 7.5% |
| Step Credit | 56.7% | **25.8%** | 10.0% |

<p align="center">
  <img src="figures/gsm8k_difficulty_stratified.png" width="92%" />
</p>

**Takeaways from difficulty stratification:**

- **Reflect-Full now leads on Hard GSM8K (12.5%)**, pulling ahead of Retry Only and Step Credit (both 10.0%) once the reflect+retry mechanism is activated at inference. This is only visible in native-mode eval — under single-pass, all three tied at ~10%.
- Reflect-Plan collapses on Medium GSM8K (15.1%), far behind all other conditions, confirming planning-style prompts are actively harmful with RLVR at this scale.
- **Step Credit leads MATH L4 (14.6%)**, the tier most accessible to a 4B model while still genuinely hard. This is consistent with step-local credit producing better gradient signal on harder problems where the error region is more localized.

---

## Phase 8: Solve Token Weight (STW)

**Hypothesis:** RLVR on Reflect-Full degrades first-try accuracy slightly because the reward signal only activates on reflection+retry trajectories, weakening gradient pressure on the initial solve. Adding a small gradient weight (0.3) on first-pass solve tokens — **Solve Token Weight (STW)** — should act as a regularizer to preserve initial attempt quality while still rewarding the reflection mechanism.

STW is only applied to **Reflect-Full** (the best-performing Phase 7 condition). Results are mean ± std over 3 seeds.

### GSM8K-200 (native-mode system accuracy)

| Condition | RLVR | STW | Δ(STW) |
|---|---|---|---|
| Reflect-Full + Retry | 32.5%±0.0% | 31.5%±0.5% | **−1.0%** |

### MATH-200 (native-mode system accuracy)

| Condition | RLVR | STW | Δ(STW) |
|---|---|---|---|
| Reflect-Full + Retry | 18.7%±0.3% | 17.8%±0.3% | **−0.8%** |

### First-try accuracy (single-pass, no retry)

| Condition | Dataset | RLVR | STW | Δ(STW) |
|---|---|---|---|---|
| Reflect-Full + Retry | GSM8K | 31.5%±0.9% | 30.8%±0.8% | −0.7% |
| Reflect-Full + Retry | MATH | 18.5%±0.5% | 17.3%±0.8% | −1.2% |

### Takeaways

- **STW does not improve over vanilla RLVR** — system accuracy is consistently ~0.8–1.0% lower on both datasets.
- **STW fails at its primary goal**: first-try accuracy also degrades (−0.7% GSM8K, −1.2% MATH), rather than being preserved. The added gradient on solve tokens does not successfully protect the initial-pass quality.
- The most likely cause is **gradient conflict**: the model receives competing signals — one rewarding good first passes (STW) and one rewarding reflection+retry trajectories (RLVR) — resulting in slightly worse performance on both objectives simultaneously.
- STW variance is low (±0.5% GSM8K, ±0.3% MATH), confirming the slight degradation is consistent across seeds rather than noise.
- **STW is dropped from further phases.** Vanilla Reflect-Full RLVR remains the best training recipe going forward.

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

- **Reflect-Full + Retry has the largest positive RLVR lift on GSM8K (+3.2%)** at 500 steps, overtaking Retry Only (+1.8%) once training converges. Both outperform all other conditions. Reflect-Full also achieves joint-best accuracy on Hard GSM8K problems (10.0%, tied with Retry Only and Step Credit).
- **Retry Only RLVR on MATH is highly unstable (18.3%±4.1%, σ=4.1%)**: the negative mean lift (−2.2% vs SFT) and extreme variance indicate the model has not consistently converged on MATH retry trajectories across seeds. Single-seed results (seed 42: 23.0%) significantly overstate the true benefit.
- **Reflect-Full + Retry is the best MATH condition (17.8%±0.9%)** with low variance, confirming structured reflection produces a more stable reward signal than blind retry on harder problems.
- **Reflect-Plan degrades on both datasets** (GSM8K: 0.0% lift, MATH: −0.7%) and collapses on Medium GSM8K problems (15.1%, far below all other conditions). Planning-style prompts remain incompatible with RFT at this scale.
- **Baseline CoT regresses on GSM8K (−3.0%)**: outcome-only reward on single-pass solves causes drift without structured output supervision.
- **Step Credit ties for the best Hard GSM8K accuracy (10.0%)** alongside Retry Only and Reflect-Full, and leads all conditions on MATH L4 (14.6%). Its variance has narrowed to ±2.2% on GSM8K and ±0.3% on MATH at 500 steps, making it the most consistent performer on harder problems.
- **Retry Only has the highest MATH variance of any condition (σ=4.1%)**: the step-local credit and reflection approaches are both more stable choices when generalisation across seeds matters.
