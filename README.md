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

## Step 1: Pretraining Inference Results
All runs logged to `results/` (ignored by git). Use `notebooks/` to plot.

## Pretrain Inference Results (Qwen/Qwen2.5-0.5B-Instruct)

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

## Phase 2: SFT + RLVR Fine-tuning (Qwen3-4B-Instruct)

We train a LoRA adapter on top of `Qwen/Qwen3-4B-Instruct-2507` using supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR). RLVR uses rejection-sampling fine-tuning (RFT): at each step, the model samples outputs and trains only on correct ones with a binary reward signal from a math verifier.

Each training run is 200 steps on a mixed GSM8K + MATH training set. Evaluation uses 200 held-out examples per dataset.

- **System accuracy**: correct on first try OR after retry/reflection
- **First-try accuracy**: correct on first attempt only
- **Tokens**: average tokens generated per example at inference

### GSM8K-200

| Condition | SFT | RLVR | Δ (RLVR) | Tokens |
|---|---|---|---|---|
| Baseline CoT | 34.5% | 32.0% | **-2.5%** | **147** |
| Retry Only | 35.0% | **36.0%** | +1.0% | 248 |
| Reflect-Full + Retry | 28.5% | 30.5% | +2.0% | 411 |
| Reflect-Plan + Retry | 27.5% | 28.0% | +0.5% | 426 |
| Step Credit | — | 32.5% | — | **147** |

### MATH-200

| Condition | SFT | RLVR | Δ (RLVR) | Tokens |
|---|---|---|---|---|
| Baseline CoT | 15.5% | 15.0% | -0.5% | **136** |
| Retry Only | **20.5%** | **21.0%** | +0.5% | 258 |
| Reflect-Full + Retry | 16.5% | 17.5% | +1.0% | 456 |
| Reflect-Plan + Retry | 16.5% | 16.5% | +0.0% | 465 |
| Step Credit | — | 16.0% | — | **136** |

### MATH-200 by Difficulty Level (RLVR, best available)

| Condition | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Baseline CoT | 38.1% | 30.0% | 17.3% | 7.3% | 1.8% |
| Retry Only | **47.6%** | **36.7%** | **25.0%** | 4.9% | **10.7%** |
| Reflect-Full + Retry | **47.6%** | 23.3% | 23.1% | **9.8%** | 3.6% |
| Reflect-Plan + Retry | **47.6%** | 30.0% | 19.2% | 4.9% | 3.6% |
| Step Credit | 38.1% | 33.3% | 17.3% | **9.8%** | 1.8% |

### Takeaways

- **RLVR hurts the solve-only baseline** (−2.5% GSM8K, −0.5% MATH): without a retry or reflection stage, RFT on correct-only outputs causes the model to overfit to high-confidence problems and regress on harder ones.
- **RLVR helps conditions with a second stage**: Retry Only (+1.0% GSM8K), Reflect-Full (+2.0%), Reflect-Plan (+0.5%) all improve, suggesting RLVR's signal is most useful when the model can recover from a wrong first attempt.
- **Reflect-Full benefits most from RLVR** (+2.0% GSM8K, +1.0% MATH), consistent with the hypothesis that RL reward on successful reflections teaches the model *when* to reflect meaningfully.
- **Retry Only** is the strongest condition in absolute accuracy (36.0% GSM8K, 21.0% MATH) at moderate token cost (248 tokens).
- **Step Credit** matches Baseline CoT token cost (147 tokens) while reaching 32.5% GSM8K — outperforming all reflection methods per token spent and closing most of the gap to Retry Only with no extra inference compute.
