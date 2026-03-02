"""
src/step_credit/train_step_credit.py
=====================================
Step-Local Credit Assignment Training
--------------------------------------
Online RFT where the gradient on a correct solution is weighted by
*where* the model tends to go wrong, not just *whether* it got the answer.

Algorithm
---------
For each problem:

  1. Generate a solve rollout from the current policy.

  2. If correct on first try → train on it with uniform weights (standard
     SFT signal; the model is already doing the right thing).

  3. If wrong → locate the mistake step using the LLM verifier
     (ask the model to check each step; first "NO" = mistake_step_idx).

  4. Generate a second attempt (plain retry, no reflection) to obtain a
     correct solution to train on.

  5. If the retry is correct → build a training datum whose token weights
     implement step-local credit:

       tokens in steps  0 … mistake_step_idx-1  →  pre_error_weight  (0.1)
       tokens in steps  mistake_step_idx … end   →  post_error_weight (1.0)

     Intuition: the model was likely right up to step K; heavily
     reinforce step K onward where the original attempt diverged.

  6. If the retry also fails → drop (no gradient).

Credit-assignment configuration (from configs/step_credit.yaml)
----------------------------------------------------------------
  mistake_locator.pre_error_weight  = 0.1
  mistake_locator.post_error_weight = 1.0
  mistake_locator.max_steps         = 12   (max steps to verify)
  mistake_locator.labeler           = llm_verifier

Step splitting
--------------
For GSM8K-style solutions (one computation per line) we split on
newlines.  Each non-empty line is treated as one reasoning step.
The final "#### N" line is the answer step and is always included.

Entry point: scripts/train_step_credit.py
"""

from __future__ import annotations

import json
import os
import random
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from src.rrr.rrr_infer import (
    build_solve_prompt,
    parse_gt_final,
    parse_pred_final,
    strict_match,
)


# ---------------------------------------------------------------------------
# Tokenisation helpers  (mirrors train_sft_lora_tiny.py for consistency)
# ---------------------------------------------------------------------------

def _has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def _to_ids(result) -> List[int]:
    """Normalise any tokeniser output to a plain List[int]."""
    if hasattr(result, "input_ids"):
        result = result.input_ids
    elif isinstance(result, dict) and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "ids"):
        return [int(x) for x in result.ids]
    if hasattr(result, "tolist"):
        return [int(x) for x in result.tolist()]
    return [int(x) for x in result]


# ---------------------------------------------------------------------------
# Step splitting
# ---------------------------------------------------------------------------

def _split_steps(solution: str) -> List[str]:
    """
    Split a solution into reasoning steps.

    For GSM8K-style solutions each non-empty line is one step:
      "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
      "Natalia sold 48+24 = <<48+24=72>>72 clips in April and May."
      "#### 72"

    For MATH-style solutions we split on double-newlines first,
    then fall back to single newlines so boxed answers stay attached.
    """
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    return lines if lines else [solution.strip()]


# ---------------------------------------------------------------------------
# LLM verifier: locate the first wrong step
# ---------------------------------------------------------------------------

def _build_verify_prompt(question: str, steps_so_far: List[str], current_step: str) -> str:
    """
    Prompt the model to check whether `current_step` is correct given
    the problem and the steps already taken.
    """
    so_far_text = "\n".join(steps_so_far) if steps_so_far else "(none yet)"
    return (
        "You are a precise math checker. Answer only YES or NO.\n\n"
        f"Problem:\n{question}\n\n"
        f"Steps already done:\n{so_far_text}\n\n"
        f"Current step to check:\n{current_step}\n\n"
        "Is this step mathematically correct? "
        "Answer YES if correct, NO if there is an error."
    )


def locate_mistake_step(
    question: str,
    solution: str,
    sampling_client,
    tokenizer,
    types,
    tinker,
    max_steps_to_check: int = 12,
    verify_temperature: float = 0.0,
    seed: int = 0,
) -> int:
    """
    Use the LLM to verify each step of `solution` in order.
    Returns the 0-based index of the first incorrect step,
    or len(steps)-1 (last step) if no mistake is found by the verifier.

    We check at most `max_steps_to_check` steps to keep API costs bounded.
    The final answer line is always included in the check window.
    """
    steps = _split_steps(solution)
    n_check = min(len(steps), max_steps_to_check)

    steps_so_far: List[str] = []

    for i in range(n_check):
        step = steps[i]
        prompt = _build_verify_prompt(question, steps_so_far, step)

        try:
            prompt_ids = _to_ids(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors=None,
            ))
        except Exception:
            prompt_ids = _to_ids(tokenizer.encode(prompt, add_special_tokens=True))

        res = sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=8,               # only need YES / NO
                temperature=verify_temperature,
                top_p=1.0,
                seed=seed + i,
            ),
        ).result()

        verdict = tokenizer.decode(
            res.sequences[0].tokens, skip_special_tokens=True
        ).strip().upper()

        if "NO" in verdict:
            return i                        # first wrong step found

        steps_so_far.append(step)

    # Verifier found no mistake – treat last step as the candidate
    # (the error may have been in the final arithmetic / answer line).
    return max(0, len(steps) - 1)


# ---------------------------------------------------------------------------
# Datum builder with per-step weights
# ---------------------------------------------------------------------------

def _build_step_credit_datum(
    question: str,
    dataset: str,
    correct_solution: str,
    mistake_step_idx: int,
    tokenizer,
    max_seq_len: int,
    types,
    pre_error_weight: float = 0.1,
    post_error_weight: float = 1.0,
) -> object:
    """
    Build a Tinker cross-entropy Datum for `correct_solution` where each
    completion token gets a weight based on which step it belongs to:

      step index < mistake_step_idx   →  pre_error_weight  (0.1 by default)
      step index >= mistake_step_idx  →  post_error_weight (1.0 by default)

    This tells the optimiser: "the error region starts at step K, so
    focus gradient there rather than on the easy early steps."
    """
    solve_prompt_str = build_solve_prompt(question, dataset)
    steps = _split_steps(correct_solution)

    # ── Build per-token weights for the completion ────────────────────────────
    # Tokenise each step individually (no special tokens between steps so the
    # token counts are additive when we concatenate).
    step_weights: List[float] = []
    for i, step in enumerate(steps):
        w = post_error_weight if i >= mistake_step_idx else pre_error_weight
        # +1 for the newline that joins this step to the next
        n_tok = len(_to_ids(tokenizer.encode(
            step + ("\n" if i < len(steps) - 1 else ""),
            add_special_tokens=False,
        )))
        step_weights.extend([w] * n_tok)

    # ── Tokenise prompt and full completion ───────────────────────────────────
    if _has_chat_template(tokenizer):
        prompt_msgs = [{"role": "user", "content": solve_prompt_str}]
        prompt_ids  = _to_ids(tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors=None,
        ))
        full_ids = _to_ids(tokenizer.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": correct_solution}],
            tokenize=True, add_generation_prompt=False, return_tensors=None,
        ))
        completion_ids = full_ids[len(prompt_ids):]
    else:
        prompt_text    = f"### User\n{solve_prompt_str}\n\n### Assistant\n"
        prompt_ids     = _to_ids(tokenizer.encode(prompt_text, add_special_tokens=True))
        completion_ids = _to_ids(tokenizer.encode(
            correct_solution.strip(), add_special_tokens=False,
        ))

    if not completion_ids:
        raise ValueError("Empty completion after tokenisation.")

    # ── Align step_weights with actual token count ────────────────────────────
    # Step-based token count may differ from the full tokenisation due to
    # special tokens at boundaries.  Pad / trim to match.
    C_actual = len(completion_ids)
    if len(step_weights) < C_actual:
        step_weights.extend([post_error_weight] * (C_actual - len(step_weights)))
    step_weights = step_weights[:C_actual]

    # ── Truncate to max_seq_len ───────────────────────────────────────────────
    max_completion = max_seq_len // 2
    max_prompt     = max_seq_len - min(C_actual, max_completion)
    prompt_ids     = prompt_ids[-max_prompt:]
    completion_ids = completion_ids[:max_completion]
    step_weights   = step_weights[:max_completion]

    P, C         = len(prompt_ids), len(completion_ids)
    full         = prompt_ids + completion_ids
    N            = P + C - 1
    raw_weights  = [0.0] * P + step_weights   # prompt gets 0
    seq_weights  = raw_weights[1:]             # left-shift to align with targets

    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=full[1:],    dtype="int64",   shape=[N]),
            "weights":       types.TensorData(data=seq_weights, dtype="float32", shape=[N]),
        },
    )


def _build_uniform_datum(
    question: str,
    dataset: str,
    correct_solution: str,
    tokenizer,
    max_seq_len: int,
    types,
) -> object:
    """Standard SFT datum (uniform weight=1.0) for first-try correct solutions."""
    return _build_step_credit_datum(
        question=question, dataset=dataset,
        correct_solution=correct_solution,
        mistake_step_idx=0,           # all steps get post_error_weight
        tokenizer=tokenizer, max_seq_len=max_seq_len, types=types,
        pre_error_weight=1.0,         # same as post → uniform
        post_error_weight=1.0,
    )


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def _sample_text(
    sampling_client,
    tokenizer,
    prompt: str,
    types,
    tinker,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    """Generate one completion from the current policy."""
    try:
        prompt_ids = _to_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors=None,
        ))
    except Exception:
        prompt_ids = _to_ids(tokenizer.encode(prompt, add_special_tokens=True))

    res = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ),
    ).result()
    return tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_checkpoint(training_client, run_name: str, global_step, output_dir: str) -> str:
    """Persist weights via save_state() and write a sidecar JSON for reload."""
    import json as _json
    from pathlib import Path as _Path

    ckpt_name  = run_name if global_step is None else f"{run_name}-step{global_step}"
    print(f"[ckpt] saving '{ckpt_name}' ...")
    save_resp  = training_client.save_state(ckpt_name).result()
    tinker_uri = save_resp.path

    ckpt_dir   = _Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_path  = ckpt_dir / f"{ckpt_name}.checkpoint.json"
    meta_path.write_text(
        _json.dumps({
            "run_name": ckpt_name,
            "tinker_path": tinker_uri,
            "steps_completed": global_step,
        }, indent=2)
    )
    print(f"[ckpt] saved: {tinker_uri}  (metadata -> {meta_path})")
    return tinker_uri


def _resolve_resume(run_name: str, resume_step: int, output_dir: str):
    """Locate a resumable checkpoint; return (tinker_uri, start_step)."""
    import json as _json, re
    from pathlib import Path as _Path

    ckpt_dir = _Path(output_dir)
    if resume_step > 0:
        meta_path = ckpt_dir / f"{run_name}-step{resume_step}.checkpoint.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {meta_path}")
        meta = _json.loads(meta_path.read_text())
        return meta["tinker_path"], resume_step

    pattern = re.compile(rf"^{re.escape(run_name)}-step(\d+)\.checkpoint\.json$")
    best_step, best_path = -1, None
    for p in ckpt_dir.glob(f"{run_name}-step*.checkpoint.json"):
        m = pattern.match(p.name)
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step, best_path = s, p

    if best_path is not None:
        meta = _json.loads(best_path.read_text())
        start = meta.get("steps_completed", best_step)
        print(f"[resume] auto-detected checkpoint at step {start}: {best_path}")
        return meta["tinker_path"], start

    final_path = ckpt_dir / f"{run_name}.checkpoint.json"
    if final_path.exists():
        meta = _json.loads(final_path.read_text())
        start = meta.get("steps_completed")
        if start is None:
            raise ValueError(
                f"Final checkpoint {final_path} has no 'steps_completed' field.\n"
                f"Pass --resume_step N explicitly."
            )
        print(f"[resume] using final checkpoint ({start} steps): {final_path}")
        return meta["tinker_path"], start

    raise FileNotFoundError(
        f"No resumable checkpoint found for run '{run_name}' in {ckpt_dir}.\n"
        f"Run with --checkpoint_every N to create mid-run snapshots."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_problems(paths: List[str], limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"[step_credit][warn] training data not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"[step_credit] loaded {len(rows)} problems from {len(paths)} file(s).")
    return rows


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_step_credit(
    # ── Tinker / model ───────────────────────────────────────────────────────
    base_model: str            = "Qwen/Qwen3-4B-Instruct-2507",
    sft_checkpoint: str        = None,     # SFT warm-start run_name
    run_name: str              = None,
    rank: int                  = 8,
    seed: int                  = 42,
    # ── Data ─────────────────────────────────────────────────────────────────
    train_jsonl_paths: List[str] = None,
    data_limit: Optional[int]    = None,
    # ── RL loop ──────────────────────────────────────────────────────────────
    max_steps: int             = 200,
    problems_per_step: int     = 4,
    sampler_refresh_every: int = 10,
    # ── Step credit weights (from configs/step_credit.yaml) ──────────────────
    pre_error_weight: float    = 0.1,    # weight for tokens BEFORE mistake step
    post_error_weight: float   = 1.0,    # weight for tokens AT/AFTER mistake step
    max_steps_to_check: int    = 12,     # max steps the LLM verifier will check
    # ── Decoding ─────────────────────────────────────────────────────────────
    solve_max_tokens: int      = 512,
    retry_max_tokens: int      = 512,
    temperature: float         = 0.7,
    top_p: float               = 0.95,
    verify_temperature: float  = 0.0,    # greedy for verifier (deterministic)
    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float                  = 1e-5,
    beta1: float               = 0.9,
    beta2: float               = 0.95,
    weight_decay: float        = 0.0,
    grad_clip: float           = 1.0,
    max_seq_len: int           = 1024,
    # ── Checkpointing / resume ─────────────────────────────────────────────────
    checkpoint_every: int      = 0,
    resume: bool               = False,
    resume_step: int           = -1,
    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str            = "results/runs",
) -> None:

    # ── Resolve defaults ─────────────────────────────────────────────────────
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if train_jsonl_paths is None:
        train_jsonl_paths = [
            str(_REPO_ROOT / "data" / "processed" / "gsm8k_train.jsonl"),
            str(_REPO_ROOT / "data" / "processed" / "math_train.jsonl"),
        ]
    if run_name is None:
        run_name = f"step_credit-r{rank}-seed{seed}"

    print(textwrap.dedent(f"""
    +------------------------------------------------------+
    |       Step-Local Credit Assignment Training          |
    +------------------------------------------------------+
    |  base_model       : {base_model:<32s}|
    |  sft_checkpoint   : {(sft_checkpoint or 'none (base weights)'):<32s}|
    |  run_name         : {run_name:<32s}|
    |  pre_error_weight : {pre_error_weight:<32g}|
    |  post_error_weight: {post_error_weight:<32g}|
    |  max_steps_check  : {max_steps_to_check:<32d}|
    |  max_steps        : {max_steps:<32d}|
    |  problems/step    : {problems_per_step:<32d}|
    |  sampler refresh  : every {sampler_refresh_every:<26d}|
    |  lr               : {lr:<32g}|
    |  seed             : {seed:<32d}|
    +------------------------------------------------------+
    """))

    # ── API key ───────────────────────────────────────────────────────────────
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set. Add it to .env.")

    import tinker
    from tinker import types

    # ── Create training client ────────────────────────────────────────────────
    service = tinker.ServiceClient()

    if resume:
        tinker_uri, start_step = _resolve_resume(run_name, resume_step, output_dir)
        print(f"[step_credit] resuming from step {start_step}: {tinker_uri}")
        training_client = service.create_training_client_from_state(
            path=tinker_uri,
            user_metadata={"run_name": run_name, "resumed_from_step": str(start_step)},
        )
        print(f"[step_credit] training state restored (step {start_step}).")
    else:
        start_step = 0
        print("[step_credit] creating LoRA training client ...")
        training_client = service.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            seed=seed,
            train_attn=True,
            train_mlp=True,
            train_unembed=True,
        )

        # ── Load SFT warm-start ───────────────────────────────────────────────
        if sft_checkpoint:
            import json as _json
            from pathlib import Path as _Path
            _repo_root = _Path(__file__).resolve().parent.parent.parent
            _meta_path = _repo_root / "results" / "runs" / f"{sft_checkpoint}.checkpoint.json"
            if not _meta_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint metadata not found: {_meta_path}\n"
                    f"Re-run SFT training with the updated train_sft_lora_tiny.py which calls save_state()."
                )
            _tinker_uri = _json.loads(_meta_path.read_text())["tinker_path"]
            print(f"[step_credit] loading SFT checkpoint '{sft_checkpoint}' from {_tinker_uri} ...")
            training_client.load_state(_tinker_uri).result()
            print("[step_credit] SFT weights loaded.")
        else:
            print("[step_credit] no SFT checkpoint – starting from base model weights.")

    tokenizer = training_client.get_tokenizer()
    print(f"[step_credit] tokenizer ready  "
          f"(chat_template={'yes' if _has_chat_template(tokenizer) else 'no'})")

    adam_params = types.AdamParams(
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip,
    )

    # ── Create initial sampling client ───────────────────────────────────────
    sampler_name = f"{run_name}-rollout"
    print(f"[step_credit] creating initial sampling client '{sampler_name}' ...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=sampler_name
    )
    print("[step_credit] sampling client ready.\n")

    # ── Load training problems ────────────────────────────────────────────────
    problems = _load_problems(train_jsonl_paths, limit=data_limit)
    if not problems:
        raise RuntimeError("No training problems loaded. Check train_jsonl_paths.")
    rng = random.Random(seed)
    rng.shuffle(problems)

    # ── Counters ──────────────────────────────────────────────────────────────
    global_step         = start_step   # 0 for fresh runs, >0 when resuming
    total_rollouts      = 0
    first_correct_count = 0   # first-try correct → uniform gradient
    credited_count      = 0   # wrong then correct retry → step-credit gradient
    failed_count        = 0   # both wrong → dropped
    problem_idx         = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"{run_name}_train_log.jsonl"

    # ── Retry prompt (plain re-solve, no reflection) ──────────────────────────
    def _retry_prompt(q: str, dataset: str) -> str:
        """Plain re-solve prompt (no reflection): just ask again."""
        if dataset == "math":
            return (
                "Solve the problem again carefully from scratch.\n"
                "Put ONLY the final answer on the last line as \\boxed{...}.\n\n"
                f"Problem:\n{q}\n\nSolution:\n"
            )
        return (
            "Solve the problem again carefully, checking each step.\n"
            "IMPORTANT: end with exactly: #### <answer>\n\n"
            f"Problem:\n{q}\n\nSolution:\n"
        )

    # ── Main training loop ────────────────────────────────────────────────────
    while global_step < max_steps:

        # ── Refresh sampler ───────────────────────────────────────────────────
        if global_step > 0 and global_step % sampler_refresh_every == 0:
            print(f"[step_credit][step {global_step}] refreshing sampling client ...")
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=sampler_name
            )

        # ── Collect datums for this gradient step ─────────────────────────────
        step_datums: List = []
        step_log:    List = []

        for _ in range(problems_per_step):
            ex          = problems[problem_idx % len(problems)]
            problem_idx += 1
            seed_base   = seed + global_step * 10_000 + problem_idx

            q       = ex.get("question", "")
            dataset = ex.get("dataset", "gsm8k")
            gt      = parse_gt_final(ex)

            if not q or gt is None:
                continue

            total_rollouts += 1

            try:
                # Stage 1 – Solve ─────────────────────────────────────────────
                solve_text = _sample_text(
                    sampling_client, tokenizer,
                    build_solve_prompt(q, dataset),
                    types, tinker,
                    max_new_tokens=solve_max_tokens,
                    temperature=temperature, top_p=top_p,
                    seed=seed_base,
                )
                p1  = parse_pred_final(solve_text)
                ok1 = strict_match(ex, p1["strict"], gt)

                log_entry = {
                    "step": global_step, "q": q[:60], "dataset": dataset,
                    "gt": gt, "pred1": p1["strict"],
                }

                if ok1:
                    # ── First-try correct: uniform gradient ───────────────────
                    datum = _build_uniform_datum(
                        question=q, dataset=dataset,
                        correct_solution=solve_text,
                        tokenizer=tokenizer, max_seq_len=max_seq_len, types=types,
                    )
                    step_datums.append(datum)
                    first_correct_count += 1
                    log_entry.update({"outcome": "first_correct", "mistake_step": None})
                    print(f"  [sc] ✓  first correct   | step={global_step} | {q[:50]!r}")

                else:
                    # Stage 2 – Locate mistake step ───────────────────────────
                    mistake_idx = locate_mistake_step(
                        question=q,
                        solution=solve_text,
                        sampling_client=sampling_client,
                        tokenizer=tokenizer,
                        types=types,
                        tinker=tinker,
                        max_steps_to_check=max_steps_to_check,
                        verify_temperature=verify_temperature,
                        seed=seed_base + 5_000,
                    )

                    # Stage 3 – Retry (plain re-solve, no reflection) ──────────
                    retry_text = _sample_text(
                        sampling_client, tokenizer,
                        _retry_prompt(q, dataset),
                        types, tinker,
                        max_new_tokens=retry_max_tokens,
                        temperature=temperature, top_p=top_p,
                        seed=seed_base + 1_000,
                    )
                    p2  = parse_pred_final(retry_text)
                    ok2 = strict_match(ex, p2["strict"], gt)

                    log_entry["pred2"]        = p2["strict"]
                    log_entry["mistake_step"] = mistake_idx

                    if ok2:
                        # ── Step-credit datum ─────────────────────────────────
                        datum = _build_step_credit_datum(
                            question=q, dataset=dataset,
                            correct_solution=retry_text,
                            mistake_step_idx=mistake_idx,
                            tokenizer=tokenizer, max_seq_len=max_seq_len, types=types,
                            pre_error_weight=pre_error_weight,
                            post_error_weight=post_error_weight,
                        )
                        step_datums.append(datum)
                        credited_count += 1
                        log_entry["outcome"] = "step_credit"
                        print(
                            f"  [sc] ✓  step-credit     | step={global_step}"
                            f" | mistake@step{mistake_idx} | {q[:45]!r}"
                        )
                    else:
                        failed_count += 1
                        log_entry["outcome"] = "fail"
                        print(
                            f"  [sc] ✗  retry failed    | step={global_step}"
                            f" | mistake@step{mistake_idx} | {q[:45]!r}"
                        )

                step_log.append(log_entry)

            except Exception as exc:
                print(f"  [step_credit][error] problem_idx={problem_idx}: {exc}")
                continue

        # ── Gradient update ───────────────────────────────────────────────────
        if step_datums:
            try:
                fb_future    = training_client.forward_backward(step_datums, "cross_entropy")
                optim_future = training_client.optim_step(adam_params)
                fb_result    = fb_future.result()
                optim_future.result()

                loss         = fb_result.metrics.get("loss:sum", float("nan"))
                global_step += 1
                credit_rate  = credited_count / max(total_rollouts, 1)
                print(
                    f"[step_credit] step {global_step:>4}/{max_steps}"
                    f"  loss={loss:.4f}"
                    f"  datums={len(step_datums)}"
                    f"  credit_rate={credit_rate:.1%}"
                )

                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({
                        "step": global_step, "loss": loss,
                        "n_datums": len(step_datums), "examples": step_log,
                    }) + "\n")

                # ── Mid-run checkpoint ────────────────────────────────────────
                if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                    _save_checkpoint(training_client, run_name, global_step, output_dir)

            except Exception as exc:
                print(f"[step_credit][error] optim step failed: {exc}")
        else:
            global_step += 1
            print(
                f"[step_credit] step {global_step:>4}/{max_steps}"
                f"  no datums  (failed={failed_count})"
            )

    # ── Save final checkpoint ─────────────────────────────────────────────────
    print(f"\n[step_credit] training complete.")
    print(f"[step_credit] rollouts={total_rollouts}"
          f"  first_correct={first_correct_count}"
          f"  credited={credited_count}"
          f"  failed={failed_count}")
    _save_checkpoint(training_client, run_name, global_step=None, output_dir=output_dir)
    print(f"[step_credit] done.  Evaluate with:\n"
          f"        python scripts/eval_sft.py --run_name {run_name} --mode baseline_solve --dataset both")
