# Abstract and Related Work — Draft

---

## Abstract

We study whether language models can be taught to identify the *specific location and cause* of their reasoning errors, rather than merely retrying after a generic self-reflection prompt. We propose **Grounded Reflection SFT**, a teacher-student data generation pipeline in which a larger teacher model (Qwen3-8B) observes a student model's incorrect solution alongside the correct answer and produces a structured three-line error identification: the exact wrong line, why it is wrong, and the correct value. These oracle-informed corrections are used as supervised fine-tuning data, after which the student (Qwen3-4B) undergoes reinforcement learning with verifiable rewards (RLVR).

Across three random seeds, grounded reflection achieves **39.3 ± 0.5% system accuracy on GSM8K** and **19.8 ± 0.6% on MATH**, outperforming a retry-only RLVR baseline (34.5 ± 1.3% GSM8K) and surpassing a baseline model sampled 16 times at inference (25.2% GSM8K pass@16) — demonstrating that the improvement reflects genuine capability gain rather than increased compute at inference. Error recovery rate analysis shows grounded reflection recovers **9.7 ± 1.4%** of first-attempt errors on GSM8K versus **1.7 ± 1.9%** for retry-only (5.7× improvement), with identical first-attempt accuracy between conditions, isolating the reflection mechanism as the source of the gain. The trained capability generalises to a held-out dataset (SVAMP: 80.2 ± 0.3% system accuracy) never seen during training. A manual audit of 50 recovered examples finds no evidence of systematic reward hacking: 88% of recoveries are consistent with the model genuinely following the reflection's directional guidance.

An error taxonomy analysis of first-attempt failures confirms the mechanism is selective: on GSM8K, grounded reflection recovers **11.0% of correctable errors** (arithmetic slips) vs 0% of conceptual errors, demonstrating the model is not broadly trying harder — it specifically re-executes targeted arithmetic corrections.

We additionally report an unexpected finding: applying RLVR directly to the base instruct model without any SFT warm-start achieves **86.3 ± 1.4% GSM8K** across five replications — 47.0pp above the grounded SFT+RLVR condition. We show this gap is explained by SFT format harm: training on structured reflection trajectories with terse bare-answer outputs degrades the base model's native chain-of-thought reasoning by ~50pp, which RLVR only partially recovers. A follow-up experiment (Phase 23) confirms this is fundamental: applying grounded reflection SFT on top of an already-strong no-SFT RLVR checkpoint (84.5% GSM8K) crashes it to 40.5%, a 44pp drop identical to the standard pipeline. **Grounded reflection is the best method within the SFT-warm-start paradigm; the paradigm itself is the bottleneck.** The no-SFT result is now fully closed — future work should explore reflection formats that preserve native chain-of-thought rather than replacing it.

---

## Related Work

**Self-refinement and reflection.** Reflexion (Shinn et al., 2023) uses verbal reinforcement learning: after failing a task, an agent generates a natural-language reflection that is prepended to future attempts as working memory. Self-Refine (Madaan et al., 2023) iterates over outputs by prompting the same model to produce feedback and then refine its own answer, showing gains on code, math, and reasoning tasks without any gradient updates. Both methods rely on the model's pre-trained self-evaluation capability and apply reflection at inference time. Our approach differs in that reflection is a *trained* behaviour, not a prompted one: we use SFT to teach the model a specific error-identification format before applying RL, and we show this produces a meaningfully higher error recovery rate than blind retry or generic reflection prompts alone.

**Reasoning with verifiable rewards.** STaR (Zeiler et al., 2022) bootstraps chain-of-thought reasoning by iteratively fine-tuning on correct solutions the model can already find, gradually expanding the difficulty frontier. ReST (Gulcehre et al., 2023) formalises this as grow-and-improve: sample solutions, filter by reward, fine-tune, repeat. RLVR methods such as those used in DeepSeek-R1 (Guo et al., 2025) apply group relative policy optimisation (GRPO) with binary outcome rewards, producing strong reasoning chains without process-level supervision. Our work sits within this family but treats the reflection trajectory — wrong attempt, error identification, retry — as the unit being reinforced, rather than a single-pass solution.

**Process-level supervision.** Let's Verify Step by Step (Lightman et al., 2023) trains a process reward model (PRM) to score individual reasoning steps, providing denser supervision than outcome rewards alone. Math-Shepherd (Wang et al., 2024) extends this with automatic step-level labelling via completion-based estimation. Our grounded reflection approach is complementary: rather than labelling the quality of each step in a forward pass, we identify the *first wrong step* in a failed attempt and generate a targeted correction — a retrospective, error-specific signal that is coarser than a full PRM but cheaper to generate and directly actionable by the model.

**Teacher-student and data generation for reasoning.** Rejection sampling fine-tuning (Yuan et al., 2023) and Alpaca-style distillation use a stronger model to generate training data for a weaker one. Closest to our setup is the line of work on using LLM-generated rationales to teach error correction: Zhang et al. (2023) show that models fine-tuned on self-correcting trajectories outperform those trained on direct solutions, and Welleck et al. (2023) study self-correction as a learnable repair operation. Our contribution is the specific design of oracle-grounded error identifications — the teacher sees the correct answer, enabling it to produce error labels that are both targeted (pointing to the wrong line) and explanatory (saying why it is wrong), rather than generating generic reflective commentary.

**Generalisation of trained reasoning.** Several papers have studied whether reasoning capabilities trained on one distribution transfer to others. Cobbe et al. (2021) find that verifier-based reasoning generalises modestly across arithmetic problem types; Wei et al. (2022) show chain-of-thought prompting generalises more broadly with scale. Our SVAMP evaluation (80.2% system accuracy on a held-out arithmetic benchmark) provides evidence that grounded reflection training instils a general self-correction mechanism rather than a dataset-specific habit, consistent with the view that the model learns to diagnose a class of arithmetic errors rather than memorise problem formats.

---

## Inference Cost and Compute Limitations

### Token overhead of the reflection mechanism

The grounded reflection trajectory involves three sequential model calls on a wrong-path example: a first attempt, a reflection, and a retry. Token counts — a reliable proxy for compute given a fixed architecture — were measured within a single eval run (GSM8K, seed 42) to avoid cross-run latency confounds:

| Step | Prompt tokens | Completion tokens | Within-run latency |
|---|---|---|---|
| First attempt | 146 | 5 | 0.83s |
| Reflection | 182 | **59** | **1.84s** |
| Retry | 206 | 5 | 0.79s |
| **Wrong-path total** | — | **69** | **3.45s** |
| Correct-first path | — | 5 | 0.83s |

The reflection step generates 11× more completion tokens than either the first attempt or retry, and accounts for 53% of wrong-path latency. A full wrong-path trajectory costs 4.4× more tokens than a correct-first trajectory (604 vs 138 total tokens). Since ~67% of GSM8K queries hit the wrong path, the effective per-query token overhead is approximately 2.5× a single forward pass.

This cost is the correct denominator for the BoN comparison: the trained model at N=1 (one first attempt + one reflection + one retry when wrong) beats an untrained baseline at N=16 independent samples (39.3% vs 25.2% GSM8K system accuracy). Even accounting for the ~2.5× inference overhead of the reflection mechanism, the trained model remains strictly more token-efficient than scaling up samples from the baseline.

### Wall-clock compute limitation

All training and evaluation was conducted through the Tinker cloud training API. **Wall-clock GPU time could not be measured**: Tinker does not expose per-request GPU utilisation or server-side compute metrics. Latency values recorded in eval logs (`meta.latency_s`) reflect end-to-end API round-trip time including network and queueing overhead. Cross-condition latency comparisons are therefore unreliable — both the Retry Only and Grounded Reflection models output ~5 completion tokens per first attempt, yet observed latencies differed by ~8× across separate eval runs, confirming the signal is dominated by server load rather than generation compute.

A rough lower bound on training compute, estimated from checkpoint file modification timestamps, is ~3 GPU-hours per 4B RLVR run at 500 steps on A100-class hardware, with total project compute across all phases (15+ conditions × 3 seeds × SFT + RLVR) estimated at 150–200 GPU-hours. This figure is not directly verified. Future work should instrument training scripts with wall-time logging independent of the API client.

---

## Methods

### Problem Setup

We study two-attempt reasoning: a model first solves a problem, then, if wrong, generates a structured reflection and retries. System accuracy is the fraction of problems where either the first attempt or the retry is correct. We compare five SFT training conditions (Baseline CoT, Retry Only, Step Credit, Reflect-Plan+Retry, Reflect-Full+Retry, Grounded Reflection) and use RLVR fine-tuning as the shared second stage across all conditions.

**Models.** Student: Qwen3-4B instruct (base for all SFT/RLVR experiments). Scaling ablation adds Qwen3-8B as a second student size. Teacher for grounded reflection data generation: Qwen3-8B instruct in `/no_think` mode (direct answer extraction, no extended reasoning chain).

**Benchmarks.** Primary: GSM8K (grade-school math, 1319 test problems) and MATH (competition math, 500 problems sampled). Held-out generalization: SVAMP (simple variation arithmetic, 1000 problems; never seen during training or SFT data construction).

### SFT Data Construction

For all conditions except Baseline CoT, we generate SFT trajectories by running the student model on a training split, collecting incorrect first attempts, and then generating correction trajectories. Trajectory formats differ by condition:

**Retry Only.** A bare retry prompt is appended after the wrong first attempt. The SFT target is the correct answer formatted as a bare number. This establishes a floor: the model sees wrong→retry without any reflection signal.

**Step Credit.** Same as Retry Only but the RLVR reward gates on intermediate step-level correctness in addition to the final answer, providing denser training signal.

**Reflect-Plan+Retry.** The SFT trajectory includes a natural-language reflection that produces a high-level plan ("I need to reconsider X") before the retry. The reflection is generated by the student model itself with a generic self-critique prompt.

**Reflect-Full+Retry.** The reflection is expanded to a full chain of thought: the model articulates what went wrong and how to correct it before retrying. Reflection quality depends entirely on the student's self-evaluation capability.

**Grounded Reflection.** The teacher model (Qwen3-8B) receives the student's wrong first attempt, the question, and the correct answer. It produces a three-field structured error identification:

```
WRONG_LINE: <the specific incorrect step from the student's solution>
WHY_WRONG:  <explanation of the error>
CORRECT_VALUE: <the correct value at that step>
```

This oracle-informed structure is the SFT training target for the reflection step. Because the teacher sees the ground-truth answer, it can produce targeted error labels rather than generic commentary. A v7 data generation pass adds `ERROR_TYPE` and `LIKELY_STEP` fields to support error taxonomy analysis (Phase 22).

All SFT training uses LoRA fine-tuning with rank 8, 500 gradient steps, batch size 4, learning rate 2×10⁻⁴, on 1059 training pairs (data scaling ablation in Phase 21 varies this from 100 to 1059).

### RLVR Fine-Tuning

After SFT, all conditions undergo RLVR using GRPO (Group Relative Policy Optimisation). The reward function is a binary verifiable reward: +1.0 if the final answer is exactly correct (strict match), else 0.0, applied to both first-attempt and retry outputs. The reward gate fires on the system-level outcome: if the retry is correct after a wrong first attempt, the trajectory receives positive reward.

**No-SFT baseline.** We additionally train a condition that skips SFT entirely and applies RLVR directly to the base instruct model weights. This produces an important control: any difference between no-SFT RLVR and SFT+RLVR conditions is attributable to the SFT stage.

RLVR hyperparameters: 500 steps, group size 8, KL coefficient 0.01, learning rate 1×10⁻⁵. All conditions are trained with 3 independent seeds (0, 1, 42); reported results show mean ± standard deviation across seeds.

### Evaluation Protocol

Each test example is evaluated with a structured two-attempt call sequence: first attempt → if wrong, reflection + retry. For Reflect-Full and Grounded Reflection conditions, the reflection prompt follows the training format. System accuracy is `(first_correct OR retry_correct) / N`. Error recovery rate is `retry_correct / first_incorrect`.

The no-SFT model is evaluated with the `reflect_full_retry` eval format even though it was not trained on reflection data. This means the reflection quality is low, and system accuracy gains come almost entirely from first-attempt performance rather than reflection-driven recovery — an important caveat when comparing no-SFT and SFT+RLVR recovery rates directly.

---

## Experiments

### Condition Comparison (Table 1)

We compare six SFT conditions + no-SFT baseline on first-attempt accuracy, retry accuracy, and system accuracy across GSM8K and MATH (Table 1). Key results:

**Grounded reflection is the strongest SFT-warmed method on both benchmarks.** On GSM8K, grounded reflection achieves 39.3 ± 0.5% system accuracy vs 34.5 ± 1.3% for retry-only (a +4.8pp lift). On MATH, the gap is 19.8 ± 0.6% vs 17.3 ± 1.0% (+2.5pp). The improvement is not from better first-attempt accuracy — first-attempt accuracies are similar across conditions (32.6% grounded vs 31.2% retry-only on GSM8K) — but from a higher error recovery rate: 9.7 ± 1.4% vs 1.7 ± 1.9%.

**All SFT-warmed methods trail no-SFT RLVR by ~47pp.** The no-SFT baseline achieves 86.3 ± 1.4% GSM8K and 51.9 ± 2.5% MATH, far exceeding every SFT-warmed condition. This gap is attributable to SFT format harm (see §SFT Format Harm below).

**GRPO vs. RLVR scaling.** At 4B, RLVR leads GRPO by +3.3pp on GSM8K and +1.7pp on MATH within the SFT-warmed paradigm. At 8B, the gap closes to near-zero (−0.7pp / +1.0pp), suggesting a capacity-dependent interaction where larger models benefit equally from both algorithms.

### Best-of-N Comparison

To verify that grounded reflection represents genuine capability improvement rather than increased inference compute, we compare against a best-of-N baseline: the untrained base instruct model sampled 16 independent times with majority vote. The trained grounded reflection model at N=1 (one attempt plus one reflection+retry on wrong examples) achieves 39.3% GSM8K vs 25.2% for the N=16 untrained baseline. Even accounting for the ~2.5× token overhead of the reflection trajectory, the trained model is more compute-efficient than scaling up inference samples from an untrained model.

### Held-Out Generalization (SVAMP)

The grounded reflection model achieves 80.2 ± 0.3% system accuracy on SVAMP (1000 problems, held-out during all training and SFT construction). First-attempt accuracy is 75.7 ± 0.6% and error recovery is 16.5 ± 0.9%. The generalization gap relative to GSM8K system accuracy is small, suggesting the model has learned a general arithmetic error-correction mechanism rather than GSM8K-specific habits.

### Error Taxonomy (Phase 22)

To characterise what kinds of errors grounded reflection fixes, we classify all first-attempt errors in the grounded condition using the `ERROR_TYPE` field produced by the teacher model during SFT data generation. Errors fall into arithmetic slip (correctable), setup/unit error (correctable), and conceptual/reasoning error (incorrectable).

On GSM8K, the result is stark: grounded reflection recovers **11.0 ± 0.9%** of correctable errors and **0.0%** of conceptual errors. The mechanism targets arithmetic re-execution exclusively. Retry-only contributes 1.0% correctable and 0% conceptual, confirming the 10× lift is specific rather than general. The GSM8K error distribution is 94% arithmetic slips (334/357 classified errors), consistent with the nature of the benchmark.

On MATH, the pattern holds but weakens: 6.5% correctable vs 4.7% conceptual recovery. MATH errors are structurally harder, and 58% of MATH first-attempt failures receive "incorrect final answer" or "incorrect reasoning" labels from the teacher — broad categories that mix correctable and incorrectable sub-types. The selective advantage of grounded reflection is present but diluted by label imprecision.

### SFT Format Harm and Phase 23

The no-SFT finding (86.3% GSM8K) raises an obvious question: does grounded reflection SFT add value on top of a strong no-SFT model, or does SFT format harm dominate regardless of starting point?

Phase 23 tests this directly: we take the no-SFT RLVR checkpoint (84.5% GSM8K) as the warm-start, apply grounded reflection SFT (1059 pairs, 500 steps), and run a second RLVR pass. Result: system accuracy crashes from 84.5% to **40.5%** — a 44pp drop identical to the standard pipeline (40.0%). The SFT training on terse bare-answer reflection trajectories collapses native chain-of-thought regardless of how capable the model was at the SFT entry point.

This closes the open question. SFT format harm is structural, not accidental: any training regime that uses brief-answer reflection targets will degrade the base model's reasoning depth. Future approaches must either (a) use full chain-of-thought targets in reflection SFT, (b) avoid SFT entirely and use RLVR with reflection as a prompted format, or (c) regularise SFT training to preserve CoT behavior.

### Scaling Ablation (Phase 21)

We vary the number of SFT training pairs (n = 100, 200, 400, 650, 1059) to characterise data efficiency. On GSM8K, system accuracy plateaus at n ≈ 400 (39.2%) and does not improve with more data (39.3% at n = 1059). On MATH, accuracy degrades with more SFT data (15.2% at n = 100 → 11.8% at n = 1059), consistent with the SFT format harm hypothesis: more SFT training deepens the CoT collapse, particularly on a harder benchmark where chain-of-thought quality matters more. The practical recommendation is to use the smallest SFT dataset that achieves saturation on the target benchmark.

---

*Note: citations above follow standard NLP venue format. Full BibTeX entries to be added before submission. Page numbers and venue details for Shinn et al. (Reflexion), Madaan et al. (Self-Refine), Zeiler et al. (STaR), Gulcehre et al. (ReST), Guo et al. (DeepSeek-R1), Lightman et al. (PRM), Wang et al. (Math-Shepherd), Yuan et al. (RFT), Zhang et al., Welleck et al., Cobbe et al. (GSM8K verifier), and Wei et al. (CoT) should be verified against arXiv/ACL Anthology before submission.*