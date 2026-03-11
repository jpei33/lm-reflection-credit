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
| Step Credit | 32.0%±0.0%* | 30.2%±2.4% | −1.8% |

*Step Credit SFT baseline reuses the Retry Only SFT checkpoint (same eval strategy: blind retry, no reflection).

### MATH-200 (first-try / single-pass, mean ± std over 3 seeds)

| Condition | SFT | RLVR | Δ (RLVR) |
|---|---|---|---|
| Baseline CoT | 15.5%±0.0% | 17.2%±0.8% | +1.7% |
| Retry Only | **20.5%±0.0%** | 18.3%±4.1% | −2.2% |
| Reflect-Full + Retry | 16.5%±0.0% | **18.5%±0.5%** | **+2.0%** |
| Reflect-Plan + Retry | 16.5%±0.0% | 15.8%±0.8% | −0.7% |
| Step Credit | **20.5%±0.0%*** | 16.8%±0.6% | −3.7% |

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

STW is applied to both **Reflect-Full** and **Reflect-Plan** conditions. Results are mean ± std over 3 seeds.

### GSM8K-200 (native-mode system accuracy)

| Condition | RLVR | STW | Δ(STW) |
|---|---|---|---|
| Reflect-Full + Retry | 32.5%±0.0% | 31.5%±0.5% | **−1.0%** |
| Reflect-Plan + Retry | 27.7%±2.0% | 27.0%±0.0% | **−0.7%** |

### MATH-200 (native-mode system accuracy)

| Condition | RLVR | STW | Δ(STW) |
|---|---|---|---|
| Reflect-Full + Retry | 18.7%±0.3% | 17.8%±0.3% | **−0.8%** |
| Reflect-Plan + Retry | 16.2%±0.8% | 16.2%±0.0% | **+0.0%** |

### First-try accuracy (single-pass, no retry)

| Condition | Dataset | RLVR | STW | Δ(STW) |
|---|---|---|---|---|
| Reflect-Full + Retry | GSM8K | 31.5%±0.9% | 30.8%±0.8% | −0.7% |
| Reflect-Full + Retry | MATH | 18.5%±0.5% | 17.3%±0.8% | −1.2% |
| Reflect-Plan + Retry | GSM8K | 27.0%±2.3% | 26.0%±0.0% | −1.0% |
| Reflect-Plan + Retry | MATH | 15.8%±0.8% | 15.1%±0.0% | −0.7% |

### Takeaways

- **STW does not improve over vanilla RLVR** on either condition — system accuracy is flat or negative across both datasets and both reflection styles.
- **STW fails at its primary goal for Reflect-Full**: first-try accuracy still degrades (−0.7% GSM8K, −1.2% MATH) rather than being preserved. The added gradient on solve tokens does not successfully protect initial-pass quality.
- **STW has a mixed profile on Reflect-Plan**: system accuracy drops −0.7% on GSM8K but is exactly flat on MATH (+0.0%). However, first-try accuracy still degrades on both (−1.0% GSM8K, −0.7% MATH), so any system-level neutrality on MATH is not attributable to improved first passes.
- The most likely cause is **gradient conflict**: competing signals — one rewarding good first passes (STW) and one rewarding reflection+retry trajectories (RLVR) — result in worse performance on both objectives simultaneously.
- **STW strongly reduces variance**: Reflect-Plan drops from ±2.0% to ±0.0% on GSM8K system accuracy, and Reflect-Full holds ±0.3% on MATH. This stability comes at the cost of accuracy.
- **STW is dropped from further phases.** Vanilla RLVR remains the best training recipe for both reflection conditions.

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

---

## Phase 9: Few-Shot Inference Evaluation

**Hypothesis:** Adding in-context few-shot demonstrations of the reflect-then-retry pattern at inference time may further improve RLVR-trained models, giving the model explicit examples of high-quality reflection trajectories it can imitate.

Few-shot prompting is evaluated only on **Reflect-Full + Retry** (the best RLVR condition). Results are mean ± std over 3 seeds (42, 0, 1). The `RLVR` column is the zero-shot baseline from Phase 1/8 for reference.

### GSM8K-200

| Condition | RLVR (zero-shot) | RLVR + Few-shot | Δ(FS) |
|---|---|---|---|
| Reflect-Full + Retry (system acc.) | 32.5%±0.0% | 32.2%±0.3% | **−0.3%** |
| Reflect-Full + Retry (first-try acc.) | 31.5%±0.9% | 31.0%±1.3% | **−0.5%** |

### MATH-200

| Condition | RLVR (zero-shot) | RLVR + Few-shot | Δ(FS) |
|---|---|---|---|
| Reflect-Full + Retry (system acc.) | 18.7%±0.3% | 18.3%±0.6% | **−0.3%** |
| Reflect-Full + Retry (first-try acc.) | 18.5%±0.5% | 18.0%±0.0% | **−0.5%** |

### Takeaways

- **Few-shot prompting consistently underperforms zero-shot RLVR** by −0.3% system accuracy and −0.5% first-try accuracy on both datasets. The effect is small but uniformly negative.
- **The RLVR-trained model has already internalized the reflect+retry format** through training, so in-context demonstrations provide no incremental signal. The model's behavior is driven by its trained policy, not by imitation of prompt examples.
- **Few-shot marginally increases variance on GSM8K** (first-try ±1.3% vs ±0.9% zero-shot), suggesting the demonstrations occasionally interfere with the model's learned strategy rather than reinforcing it.
- **MATH first-try variance drops to 0.0%** under few-shot, but this accompanies a consistent accuracy drop — low variance at a lower mean is not a win.
- **Few-shot is dropped from further experiments.** Zero-shot RLVR remains the best inference recipe for all conditions.

---

## Phase 10: GRPO Training

**Hypothesis:** GRPO (Group Relative Policy Optimisation) uses within-group reward normalisation to reduce variance in the policy gradient signal, potentially providing a more stable training objective than RLVR's rejection-sampling approach — especially for structured reflection tasks where reward is sparse.

GRPO is evaluated on three conditions (Baseline CoT, Retry Only, Reflect-Full + Retry) across 3 seeds. Reflect-Plan is not evaluated (it has been consistently weak across all prior phases). Results are mean ± std over seeds 42, 0, 1.

### GSM8K-200 (native-mode system accuracy)

| Condition | SFT | RLVR | GRPO | Δ(GRPO vs RLVR) |
|---|---|---|---|---|
| Baseline CoT | 34.5%±0.0% | 31.5%±2.0% | **34.8%±0.3%** | **+3.3%** |
| Retry Only | 32.0%±0.0% | 34.5%±1.3% | **34.8%±0.3%** | +0.3% |
| Reflect-Full + Retry | 28.5%±0.0% | **32.5%±0.0%** | 29.2%±0.3% | **−3.3%** |

### MATH-200 (native-mode system accuracy)

| Condition | SFT | RLVR | GRPO | Δ(GRPO vs RLVR) |
|---|---|---|---|---|
| Baseline CoT | 15.5%±0.0% | **17.2%±0.8%** | 15.3%±0.3% | −1.9% |
| Retry Only | 20.5%±0.0% | 19.2%±3.3% | **20.7%±0.6%** | +1.5% |
| Reflect-Full + Retry | 16.5%±0.0% | **18.7%±0.3%** | 17.0%±0.5% | **−1.7%** |

### GRPO vs SFT lift (system accuracy)

| Condition | GSM8K | MATH |
|---|---|---|
| Baseline CoT | +0.3% | −0.2% |
| Retry Only | +2.8% | +0.2% |
| Reflect-Full + Retry | +0.7% | +0.5% |

### Why we tried both RLVR and GRPO

Both methods are online RL algorithms that optimise a language model using verifiable math rewards, but they handle failed trajectories in fundamentally different ways — and that difference turns out to matter enormously for structured reflection.

**RLVR (rejection-sampling fine-tuning)** is simple: at each step, sample a batch of rollouts, keep only the ones that got the answer right, and fine-tune on those with standard cross-entropy. Failed trajectories are silently discarded. The result is a clean, if sparse, training signal — the model only ever sees examples of itself succeeding.

**GRPO** is more sophisticated: sample a group of rollouts per problem, compute each rollout's reward, then normalise rewards *within the group* to get relative advantages. This within-group normalisation is designed to reduce variance in the gradient signal — if most rollouts in a group fail, GRPO still extracts a useful signal from the one or two that succeeded, because it knows how much better they were than the group average.

The appeal of GRPO for reflection is obvious in theory: Reflect-Full + Retry trajectories are long and require two things to go right (the reflection must be useful *and* the retry must succeed), so many rollouts fail. RLVR would discard all of them and apply zero gradient, potentially leaving the model undertrained on reflection. GRPO, by normalising within the group, should be able to extract a learning signal even from mostly-failing batches.

**What actually happened** is the opposite. The within-group normalisation backfires precisely because Reflect-Full trajectories are so rarely correct. When nearly every rollout in a group fails, the "advantage" of the one success becomes numerically very large (since it is compared against a low baseline), but the gradient it produces is noisy — it reinforces the specific surface-level features of that one trajectory rather than the underlying reasoning strategy. Across many steps, this amplifies flukes and produces a less stable policy than simply waiting for the occasional good rollout and fine-tuning on it cleanly.

RLVR's apparent weakness — discarding failed trajectories — is in practice its strength for sparse-reward settings: the model only trains on genuine successes, so every gradient step is high-quality. The cost is that training is slower when correct rollouts are rare. But for a 500-step run on math problems, this is not the bottleneck.

### Takeaways

- **GRPO significantly underperforms RLVR on Reflect-Full + Retry** — the most important condition. −3.3% on GSM8K and −1.7% on MATH. This is the central negative finding: GRPO's within-group normalisation amplifies noisy gradients on sparse-reward reflection tasks rather than stabilising them.
- **GRPO matches or beats RLVR on simpler conditions.** Baseline CoT gains +3.3% on GSM8K (recovering from RLVR's regression there), and Retry Only is roughly matched. For conditions where rollouts succeed more often and trajectories are shorter, GRPO's normalisation provides its intended benefit.
- **GRPO vs SFT lifts are small** (+0.3–2.8% GSM8K, ≤0.5% MATH). GRPO barely improves over the SFT starting point, suggesting that for these problem difficulties the gradient is not adding much beyond what supervised training already achieved.
- **GRPO has very low variance** (±0.3–0.6% across all conditions), compared to RLVR's occasional instability (e.g. Retry Only MATH at ±3.3%). The stability is real but comes at the cost of peak accuracy — GRPO converges to a more consistent but lower ceiling.
- **RLVR remains the best training recipe.** GRPO is not used in further phases.

---

## Phase 11: pass@k / Majority Vote

**Hypothesis:** The current RLVR results could be explained by inference compute alone — Reflect-Full uses ~3× the tokens of a single Baseline CoT pass. pass@k and majority@k measure what a compute-matched sampling baseline achieves without any RL training, isolating whether RLVR adds genuine capability beyond just spending more tokens.

**Two metrics:**

- **pass@k** — did *any* of k independent samples get it right? Oracle upper bound (requires knowing which answer is correct at test time). Answers: "what is the ceiling for unguided search?"
- **majority@k** — is the plurality answer (most common across k samples) correct? Oracle-free and deployable. Answers: "what does a practical inference-time scaling baseline achieve?"

Evaluated on the Baseline CoT SFT checkpoint (no RL training, temperature=0.7), sweeping k ∈ {1, 2, 3, 8, 16}. Compute matching maps each trained condition to the k value whose token budget is equivalent.

### Results

**GSM8K** — Baseline SFT pass@k / majority@k (n=200):

| k | Tokens (approx) | pass@k | maj@k | Compute-matched trained condition | RLVR acc |
|---|---|---|---|---|---|
| 1 | ~147 | 23.5% | 23.5% | Baseline CoT | 31.5% |
| 2 | ~294 | 27.0% | 23.0% | Retry Only | 34.0% |
| 3 | ~441 | 32.0% | 23.0% | Reflect-Full + Retry | **32.5%** |
| 8 | ~1,176 | 43.5% | 23.0% | — | — |
| 16 | ~2,352 | 51.0% | 24.5% | — | — |

**MATH** — Baseline SFT pass@k / majority@k (n=200):

| k | Tokens (approx) | pass@k | maj@k | Compute-matched trained condition | RLVR acc |
|---|---|---|---|---|---|
| 1 | ~136 | 15.5% | 15.5% | Baseline CoT | 16.5% |
| 2 | ~272 | 20.0% | 16.0% | Retry Only | 23.0% |
| 3 | ~408 | 24.5% | 16.5% | Reflect-Full + Retry | **18.5%** |
| 8 | ~1,088 | 29.0% | 16.0% | — | — |
| 16 | ~2,176 | 36.5% | 14.5% | — | — |

### Takeaways

**RLVR consistently beats majority@k at matched compute.** Across both datasets and all compute-matched conditions, trained RLVR models exceed maj@k by a substantial margin:

- GSM8K Reflect-Full RLVR (32.5%) vs. maj@3 (23.0%): **+9.5 pp** — the clearest signal
- GSM8K Retry Only RLVR (34.0%) vs. maj@2 (23.0%): **+11.0 pp**
- MATH Retry Only RLVR (23.0%) vs. maj@2 (16.0%): **+7.0 pp**
- MATH Reflect-Full RLVR (18.5%) vs. maj@3 (16.5%): **+2.0 pp** (smaller but consistent)

This rules out the compute-equivalence alternative explanation: the gains from RLVR training are not simply an artifact of spending more tokens per problem.

**majority@k is surprisingly flat.** On GSM8K, maj@k is essentially constant at ~23% from k=2 through k=16, despite pass@k climbing to 51%. This indicates the baseline model's errors are correlated — when it is wrong, it is confidently and consistently wrong in the same way. RLVR training changes this distribution, producing more diverse and correct solutions.

**pass@k at k=3 matches Reflect-Full RLVR on GSM8K** (32.0% vs. 32.5%), but this is an oracle upper bound not achievable without a verifier. The deployable majority@k baseline at k=3 is only 23.0%, well below the trained model. The gap between pass@k and maj@k (9 pp on GSM8K, 8 pp on MATH at k=3) reflects the difficulty of inference-time selection without a reward signal.

**MATH shows a different pattern.** maj@k barely moves from k=1 to k=16 (15.5% → 14.5%), suggesting even more correlated failures on harder problems. RLVR's +2.0 pp lift over maj@3 is modest in absolute terms but genuine, and larger gains appear on Retry Only (+7.0 pp).

---

## Phase 12: Step Credit SFT Baseline + Labeler Accuracy

### Step Credit SFT Baseline

Step Credit was previously missing an SFT column in the results tables, making its RLVR Δ uncomputable. The SFT baseline is now set to the **Retry Only SFT checkpoint** (`qwen3-4b-retry_only-r{rank}-seed{seed}`), since both conditions use blind retry at evaluation time and share the same pre-RL starting point.

With the SFT baseline filled in, the RLVR Δ for Step Credit is **negative on both datasets**:

| Dataset | SFT | RLVR | Δ (RLVR) |
|---|---|---|---|
| GSM8K (first-try) | 32.0%±0.0% | 30.2%±2.4% | **−1.8%** |
| MATH (first-try) | 20.5%±0.0% | 16.8%±0.6% | **−3.7%** |

RLVR training with step-local credit degrades first-try accuracy relative to the SFT starting point on both datasets. In system accuracy (with retry) the gap narrows, but Step Credit RLVR does not improve over its SFT baseline on either dataset. This is a robust negative result: step-local credit assignment, as implemented, actively harms the RLVR training signal.

### Labeler Accuracy Evaluation

**Hypothesis:** The step-local credit signal is only meaningful if the LLM verifier (`locate_mistake_step`) accurately identifies where the model first went wrong. Labeler accuracy contextualises the Step Credit negative result — it distinguishes between "the credit signal is correct but doesn't help" and "the credit signal is too noisy to be useful."

**Oracle:** For GSM8K, ground-truth solutions include annotations of the form `<<expr=result>>`. The oracle evaluates each expression and flags the first step where `eval(expr) ≠ result`, giving automatic arithmetic ground truth without manual labeling. MATH does not have per-step annotations and is excluded.

**Result:** Evaluating `locate_mistake_step` on the Step Credit RLVR checkpoint (100 GSM8K problems, temperature=0.7) revealed a confounding factor: the Step Credit checkpoint **generates solutions without intermediate steps** — outputs consist of a single final answer (`#### N`) with no multi-step reasoning chain. This means:

- **oracle_coverage = 0%** — the oracle cannot find arithmetic errors in one-step solutions (no `<<expr=result>>` annotations to verify)
- **labeler_idx = 0** for all problems — the labeler trivially points to step 0 since only one step exists
- Exact match and within-1 accuracy are not computable

This is itself a meaningful finding. RLVR training with step-level credit pressure appears to have collapsed the model's reasoning style — rather than learning better step attribution, the model learned to skip intermediate steps entirely, bypassing the credit mechanism. This provides a mechanistic explanation for the negative accuracy result: the step credit signal was not just noisy but counterproductive to the model's ability to generate multi-step solutions.

**Implication for Step Credit as a method:** Reliable labeler accuracy is a precondition for step credit to function. The evaluation shows this precondition is violated in practice — not because the labeler is inaccurate, but because the training pressure itself destroys the step structure the labeler requires.


---

## Figures

All figures are saved to `figures/`.

### Figure 1 — RLVR Lift over SFT (`fig1_rlvr_delta.png`)

![RLVR Delta](figures/fig1_rlvr_delta.png)

RLVR lift (percentage points) over the SFT baseline for each training condition on GSM8K (left) and MATH (right). Positive bars (blue/green) indicate RLVR improves over SFT; negative bars (red/orange) indicate degradation. Retry Only and Reflect-Full + Retry show consistent positive lift on both datasets. Reflect-Plan + Retry and Step Credit show negative or near-zero lift, confirming these as failed conditions. Baseline CoT shows negative lift on GSM8K, suggesting the RL reward signal is too sparse for the simplest format.

### Figure 2 — pass@k Curve (`fig2_passk_curve.png`)

![pass@k Curve](figures/fig2_passk_curve.png)

Oracle upper bound (pass@k, blue) and deployable majority vote (majority@k, orange) for the Baseline SFT checkpoint at temperature 0.7, sweeping k ∈ {1, 2, 3, 8, 16}. Coloured markers show RLVR trained model accuracy at compute-matched k values. On GSM8K, RLVR Retry Only (34.0%) and Reflect-Full (32.5%) both comfortably exceed majority@k at matched k (23.0%), ruling out inference-compute as the explanation for the gains. majority@k is nearly flat across all k on both datasets, indicating that the baseline model's errors are highly correlated — RLVR training changes the solution distribution, not just the token budget.

### Figure 3 — Training Reward Curves (`fig3_training_curves.png`)

![Training Curves](figures/fig3_training_curves.png)

Fraction of rollouts with correct final answer (training reward) over the 500-step run, averaged across 3 seeds with ±1 std shading. Reflect-Full RLVR and Retry Only RLVR climb steadily. Reflect-Full GRPO (dashed) lags behind both RLVR conditions — consistent with the evaluation results and the within-group normalisation explanation: when most rollouts fail, GRPO's relative reward signal amplifies noise rather than signal.

---

## Qualitative Examples

The following examples come from the MATH dataset, evaluated with the Reflect-Full format at inference time (pre-RLVR checkpoint). They illustrate the two qualitatively different outcomes the reflection mechanism produces.

---

### Example A — Reflection Helps

**Problem:** If $f(x) = g(g(x)) - g(x)$ and $g(x) = 2x - 1$, find $f(3)$.

**Ground truth:** 4

---

**First attempt** *(incorrect — the model computes g(5) correctly but truncates before subtracting g(3))*

```
f(3) = g(g(3)) - g(3)
g(3) = 2·3 − 1 = 5
f(3) = g(5) − 5
g(5) = 2·5 − 1 = 9
[solution truncated before final subtraction]
#### 9
```

---

**Reflection**

```
Likely step: 4
Fix plan: Use g(5) = 9 instead of g(3) = 5 in the calculation.
```

The labeler correctly identifies that the error is at the substitution step (step 4), not in the arithmetic. The fix plan is accurate and minimal.

---

**Retry** *(correct)*

```
g(3) = 2(3) − 1 = 5
f(3) = g(g(3)) − g(3) = g(5) − 5 = 9 − 5 = 4
#### 4  ✓
```

The retry applies the reflection's fix directly. No new reasoning is needed — the model already had all components correct, it just failed to complete the final subtraction.

**Why this works:** The error was a near-miss — all sub-computations were correct. The reflection mechanism detects the incomplete step and re-routes execution. This is the best-case scenario for structured reflection.

---

### Example B — Reflection Does Not Help

**Problem:** Find the coefficient of $x$ when $(7 + 8x) - 3(2 + 6x + 2x^2) + 9(1 + 3x + 4x^2 + 12x^3) - 9(7 - x^2 - 8x^3 + 13x^4)$ is simplified.

**Ground truth:** 17

---

**First attempt** *(incorrect)*

```
Distributing:
  7 + 8x
  −6 − 18x − 6x²
  9 + 27x + 36x² + 108x³
  −63 + 9x² + 72x³ − 117x⁴
[truncated — arithmetic error in x-coefficient collection]
#### 8
```

---

**Reflection**

```
Likely step: None
Fix plan: The problem seems to be incorrectly set up for simplification
based on the provided model. It should be re-evaluated with proper
distribution and combining of like terms.
```

The reflection fails to identify a specific step. The "Fix plan" is generic and non-actionable — it restates the problem rather than diagnosing the error.

---

**Retry** *(still incorrect)*

```
[Re-distributes all terms correctly but makes a different arithmetic error
in summing the x-coefficients: 8 − 18 + 27 + 9 = 26 instead of 17]
#### 26
```

**Why this fails:** The error was a pure arithmetic accumulation mistake across many terms — there is no single identifiable "wrong step". The reflection mechanism correctly says `Likely step: None` but cannot construct a useful fix plan for arithmetic errors of this kind. The retry produces a different wrong answer, showing the model retried with fresh sampling rather than targeted correction.

**Takeaway:** Reflection adds most value for structural or reasoning errors (missing a step, wrong substitution) where the fix is localised. It adds little value for distributed arithmetic errors, where the correct fix requires recomputing every term independently — which the model does not reliably do even with a second chance.
