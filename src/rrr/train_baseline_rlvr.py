"""
src/rrr/train_baseline_rlvr.py
================================
Baseline RLVR-CoT and Retry-only RLVR Training
------------------------------------------------
Two simple RLVR conditions that complete the comparison matrix against
RRR and step-local credit assignment.

mode="solve"  —  Baseline RLVR-CoT
  The simplest possible outcome-reward training:
    1. Sample a solve rollout from the current policy.
    2. If correct → train on it with uniform weight (reward = 1).
    3. If wrong   → skip (reward = 0, no gradient).

  This is rejection-sampling fine-tuning (RFT) on first-try correct
  solutions.  It is the natural RLVR extension of the baseline_solve
  SFT condition.  Use it to answer: "does *any* RLVR signal help, vs
  just SFT behavioural cloning?"

mode="retry"  —  Retry-only RLVR
  Outcome-reward on retry without any reflection step:
    1. Sample a solve rollout.
    2. If already correct → skip (clean baseline, nothing to improve).
    3. If wrong → sample a retry (same problem, no reflection prompt).
    4. If retry correct → train on retry tokens only (solve = context,
       weight 0).
    5. If retry also wrong → skip.

  The natural RLVR extension of the retry_only SFT condition.
  Comparing this against RRR isolates the value of explicit structured
  reflection: "does adding a reflect step between solve and retry
  improve over blind retry?"

Credit-assignment summary
--------------------------
  mode    |  solve tokens  |  reflect tokens  |  retry tokens
  --------|----------------|------------------|---------------
  solve   |      1.0       |       n/a        |      n/a
  retry   |      0.0       |       n/a        |      1.0

Entry point: scripts/train_baseline_rlvr.py
"""

from __future__ import annotations

import json
import os
import random
import textwrap
from pathlib import Path
from typing import List, Optional

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
# Datum builders
# ---------------------------------------------------------------------------

def _messages_to_datum(messages, tokenizer, max_seq_len: int, types):
    """
    Build a Tinker cross-entropy Datum training on the LAST assistant turn.
    All earlier turns are context (weight 0).
    """
    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if messages[i]["role"] == "assistant"),
        None,
    )
    if last_asst is None:
        raise ValueError("No assistant message in conversation.")

    prompt_msgs = messages[:last_asst]

    if _has_chat_template(tokenizer):
        prompt_ids = _to_ids(tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors=None,
        ))
        full_ids = _to_ids(tokenizer.apply_chat_template(
            messages[:last_asst + 1], tokenize=True,
            add_generation_prompt=False, return_tensors=None,
        ))
        completion_ids = full_ids[len(prompt_ids):]
    else:
        parts = [
            f"### {m['role'].capitalize()}\n{m['content'].strip()}"
            for m in prompt_msgs
        ]
        prompt_text    = "\n\n".join(parts) + ("\n\n" if parts else "") + "### Assistant\n"
        prompt_ids     = _to_ids(tokenizer.encode(prompt_text, add_special_tokens=True))
        completion_ids = _to_ids(tokenizer.encode(
            messages[last_asst]["content"].strip(), add_special_tokens=False,
        ))

    if not completion_ids:
        raise ValueError("Empty completion after tokenisation.")

    max_completion = max_seq_len // 2
    max_prompt     = max_seq_len - min(len(completion_ids), max_completion)
    prompt_ids     = prompt_ids[-max_prompt:]
    completion_ids = completion_ids[:max_completion]

    P, C        = len(prompt_ids), len(completion_ids)
    full        = prompt_ids + completion_ids
    N           = P + C - 1
    raw_weights = [0.0] * P + [1.0] * C
    seq_weights = raw_weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=full[1:],    dtype="int64",   shape=[N]),
            "weights":       types.TensorData(data=seq_weights, dtype="float32", shape=[N]),
        },
    )


def _build_solve_datum(question, dataset, solve_text, tokenizer, max_seq_len, types):
    """
    Datum for mode='solve': train on the correct first-try solution.
    Loss on solution tokens only (prompt = weight 0).
    """
    msgs = [
        {"role": "user",      "content": build_solve_prompt(question, dataset)},
        {"role": "assistant", "content": solve_text},
    ]
    return _messages_to_datum(msgs, tokenizer, max_seq_len, types)


def _retry_prompt_str(question: str, dataset: str) -> str:
    """Plain re-solve prompt with no reflection cue."""
    if dataset == "math":
        return (
            "Solve the problem again carefully from scratch.\n"
            "Put ONLY the final answer on the last line as \\boxed{...}.\n\n"
            f"Problem:\n{question}\n\nSolution:\n"
        )
    return (
        "Solve the problem again carefully, checking each step.\n"
        "IMPORTANT: end with exactly: #### <answer>\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def _build_retry_datum(question, dataset, solve_text, retry_text, tokenizer, max_seq_len, types):
    """
    Datum for mode='retry': train on the correct retry only.
    Solve = context (weight 0), retry = target (weight 1).
    """
    msgs = [
        {"role": "user",      "content": build_solve_prompt(question, dataset)},
        {"role": "assistant", "content": solve_text},
        {"role": "user",      "content": _retry_prompt_str(question, dataset)},
        {"role": "assistant", "content": retry_text},
    ]
    return _messages_to_datum(msgs, tokenizer, max_seq_len, types)


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def _make_prompt_ids(tokenizer, prompt: str) -> list:
    """Tokenize a prompt string to a plain list of ints."""
    try:
        return _to_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors=None,
        ))
    except Exception:
        return _to_ids(tokenizer.encode(prompt, add_special_tokens=True))


def _fire_sample(sampling_client, tokenizer, prompt, types, tinker,
                 max_new_tokens, temperature, top_p, seed):
    """Fire a sample request and return the future immediately (non-blocking)."""
    prompt_ids = _make_prompt_ids(tokenizer, prompt)
    return sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ),
    )


def _decode_future(future, tokenizer) -> str:
    """Block on a sample future and decode the result to text."""
    res = future.result()
    return tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True).strip()


def _sample_text(sampling_client, tokenizer, prompt, types, tinker,
                 max_new_tokens, temperature, top_p, seed) -> str:
    """Generate one completion (fire-and-wait convenience wrapper)."""
    return _decode_future(
        _fire_sample(sampling_client, tokenizer, prompt, types, tinker,
                     max_new_tokens, temperature, top_p, seed),
        tokenizer,
    )


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_checkpoint(training_client, run_name: str, global_step, output_dir: str) -> str:
    """
    Persist current weights via save_state() and write a sidecar JSON so that
    eval_sft.py (and RLVR --sft_checkpoint) can reload them by name.

    If global_step is None, saves as the final checkpoint under run_name.
    Otherwise saves under run_name-step{global_step} as a mid-run snapshot.
    Returns the Tinker URI.
    """
    import json as _json
    from pathlib import Path as _Path

    if global_step is None:
        ckpt_name = run_name
    else:
        ckpt_name = f"{run_name}-step{global_step}"

    print(f"[ckpt] saving '{ckpt_name}' ...")
    save_resp   = training_client.save_state(ckpt_name).result()
    tinker_uri  = save_resp.path

    ckpt_dir    = _Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_path   = ckpt_dir / f"{ckpt_name}.checkpoint.json"
    meta_path.write_text(
        _json.dumps({"run_name": ckpt_name, "tinker_path": tinker_uri}, indent=2)
    )
    print(f"[ckpt] saved: {tinker_uri}  (metadata -> {meta_path})")
    return tinker_uri


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_problems(paths: List[str], limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"[baseline_rlvr][warn] not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"[baseline_rlvr] loaded {len(rows)} problems from {len(paths)} file(s).")
    return rows


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_baseline_rlvr(
    # ── Mode ─────────────────────────────────────────────────────────────────
    mode: str                  = "solve",   # "solve" | "retry"
    # ── Tinker / model ───────────────────────────────────────────────────────
    base_model: str            = "Qwen/Qwen3-4B-Instruct-2507",
    sft_checkpoint: str        = None,
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
    # ── Decoding ─────────────────────────────────────────────────────────────
    solve_max_tokens: int      = 512,
    retry_max_tokens: int      = 512,
    temperature: float         = 0.7,
    top_p: float               = 0.95,
    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float                  = 1e-5,
    beta1: float               = 0.9,
    beta2: float               = 0.95,
    weight_decay: float        = 0.0,
    grad_clip: float           = 1.0,
    max_seq_len: int           = 1024,
    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_every: int      = 0,        # save_state every N steps (0 = off)
    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str            = "results/runs",
) -> None:

    if mode not in ("solve", "retry"):
        raise ValueError(f"mode must be 'solve' or 'retry', got {mode!r}")

    # ── Resolve defaults ─────────────────────────────────────────────────────
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if train_jsonl_paths is None:
        train_jsonl_paths = [
            str(_REPO_ROOT / "data" / "processed" / "gsm8k_train.jsonl"),
            str(_REPO_ROOT / "data" / "processed" / "math_train.jsonl"),
        ]
    if run_name is None:
        run_name = f"baseline_rlvr_{mode}-r{rank}-seed{seed}"

    # Default SFT warm-start per mode
    if sft_checkpoint is None:
        sft_map = {
            "solve": f"qwen3-4b-baseline_solve-r{rank}-seed{seed}",
            "retry": f"qwen3-4b-retry_only-r{rank}-seed{seed}",
        }
        sft_checkpoint = sft_map[mode]
        print(f"[baseline_rlvr] --sft_checkpoint not set, defaulting to '{sft_checkpoint}'")

    mode_description = {
        "solve": "Outcome reward on first-try solve  (Baseline RLVR-CoT)",
        "retry": "Outcome reward on retry, no reflect (Retry-only RLVR)",
    }[mode]

    print(textwrap.dedent(f"""
    +------------------------------------------------------+
    |          Baseline RLVR Training                      |
    +------------------------------------------------------+
    |  mode             : {mode:<32s}|
    |  description      : {mode_description:<32s}|
    |  base_model       : {base_model:<32s}|
    |  sft_checkpoint   : {sft_checkpoint:<32s}|
    |  run_name         : {run_name:<32s}|
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

    # ── Training client ───────────────────────────────────────────────────────
    service = tinker.ServiceClient()
    print("[baseline_rlvr] creating LoRA training client ...")
    training_client = service.create_lora_training_client(
        base_model=base_model,
        rank=rank,
        seed=seed,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
    )

    # ── Load SFT warm-start ───────────────────────────────────────────────────
    # sft_checkpoint is a run name; resolve to the Tinker URI via the sidecar
    # file written by train_sft_lora_tiny.py after training.
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
    print(f"[baseline_rlvr] loading SFT checkpoint '{sft_checkpoint}' from {_tinker_uri} ...")
    training_client.load_state(_tinker_uri).result()
    print("[baseline_rlvr] SFT weights loaded.")

    tokenizer = training_client.get_tokenizer()
    print(f"[baseline_rlvr] tokenizer ready  "
          f"(chat_template={'yes' if _has_chat_template(tokenizer) else 'no'})")

    adam_params = types.AdamParams(
        learning_rate=lr, beta1=beta1, beta2=beta2,
        weight_decay=weight_decay, grad_clip_norm=grad_clip,
    )

    # ── Initial sampling client ───────────────────────────────────────────────
    sampler_name = f"{run_name}-rollout"
    print(f"[baseline_rlvr] creating sampling client '{sampler_name}' ...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=sampler_name
    )
    print("[baseline_rlvr] sampling client ready.\n")

    # ── Load problems ─────────────────────────────────────────────────────────
    problems = _load_problems(train_jsonl_paths, limit=data_limit)
    if not problems:
        raise RuntimeError("No training problems loaded.")
    rng = random.Random(seed)
    rng.shuffle(problems)

    # ── Counters ──────────────────────────────────────────────────────────────
    global_step    = 0
    total_rollouts = 0
    total_trained  = 0   # datums that produced a gradient
    total_skipped  = 0   # wrong (no useful signal for this mode)
    problem_idx    = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"{run_name}_train_log.jsonl"

    # ── Training loop ─────────────────────────────────────────────────────────
    while global_step < max_steps:

        # ── Refresh sampler to keep rollouts on-policy ────────────────────────
        if global_step > 0 and global_step % sampler_refresh_every == 0:
            print(f"[baseline_rlvr][step {global_step}] refreshing sampling client ...")
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=sampler_name
            )

        step_datums: List = []
        step_log:    List = []

        # ── Gather this step's problems ───────────────────────────────────────
        step_examples = []
        for _ in range(problems_per_step):
            ex        = problems[problem_idx % len(problems)]
            problem_idx += 1
            seed_base = seed + global_step * 10_000 + problem_idx
            q         = ex.get("question", "")
            dataset   = ex.get("dataset", "gsm8k")
            gt        = parse_gt_final(ex)
            if not q or gt is None:
                continue
            total_rollouts += 1
            step_examples.append((ex, q, dataset, gt, seed_base))

        # ── Stage 1: Fire all solve requests in parallel ──────────────────────
        solve_futures = [
            _fire_sample(
                sampling_client, tokenizer,
                build_solve_prompt(q, dataset),
                types, tinker,
                max_new_tokens=solve_max_tokens,
                temperature=temperature, top_p=top_p,
                seed=seed_base,
            )
            for (ex, q, dataset, gt, seed_base) in step_examples
        ]

        # ── Collect solve results ─────────────────────────────────────────────
        solve_results = []
        for (ex, q, dataset, gt, seed_base), fut in zip(step_examples, solve_futures):
            try:
                solve_text = _decode_future(fut, tokenizer)
                p1  = parse_pred_final(solve_text)
                ok1 = strict_match(ex, p1["strict"], gt)
                solve_results.append((ex, q, dataset, gt, seed_base, solve_text, p1, ok1))
            except Exception as exc:
                print(f"  [baseline_rlvr][error] solve stage: {exc}")

        if mode == "solve":
            # ── Baseline RLVR-CoT: train on correct first-try solves ──────────
            for (ex, q, dataset, gt, seed_base, solve_text, p1, ok1) in solve_results:
                log_entry = {
                    "step": global_step, "q": q[:60], "dataset": dataset,
                    "gt": gt, "pred1": p1["strict"],
                }
                if ok1:
                    datum = _build_solve_datum(
                        q, dataset, solve_text, tokenizer, max_seq_len, types
                    )
                    step_datums.append(datum)
                    total_trained += 1
                    log_entry["outcome"] = "trained"
                    print(f"  [bl] ✓  correct solve   | step={global_step} | {q[:50]!r}")
                else:
                    total_skipped += 1
                    log_entry["outcome"] = "skip_wrong"
                    print(f"  [bl] ✗  wrong, skipped  | step={global_step} | {q[:50]!r}")
                step_log.append(log_entry)

        else:  # mode == "retry"
            # ── Retry-only RLVR: log already-correct, batch-fire retries ──────
            wrong_batch = []
            for (ex, q, dataset, gt, seed_base, solve_text, p1, ok1) in solve_results:
                if ok1:
                    total_skipped += 1
                    step_log.append({
                        "step": global_step, "q": q[:60], "dataset": dataset,
                        "gt": gt, "pred1": p1["strict"],
                        "outcome": "skip_already_correct",
                    })
                    print(f"  [rt] –  already correct | step={global_step} | {q[:50]!r}")
                else:
                    wrong_batch.append((ex, q, dataset, gt, seed_base, solve_text, p1))

            # ── Stage 2: Fire all retry requests in parallel ──────────────────
            retry_futures = [
                _fire_sample(
                    sampling_client, tokenizer,
                    _retry_prompt_str(q, dataset),
                    types, tinker,
                    max_new_tokens=retry_max_tokens,
                    temperature=temperature, top_p=top_p,
                    seed=seed_base + 1_000,
                )
                for (ex, q, dataset, gt, seed_base, solve_text, p1) in wrong_batch
            ]

            for (ex, q, dataset, gt, seed_base, solve_text, p1), fut in zip(wrong_batch, retry_futures):
                try:
                    retry_text = _decode_future(fut, tokenizer)
                    p2  = parse_pred_final(retry_text)
                    ok2 = strict_match(ex, p2["strict"], gt)
                    log_entry = {
                        "step": global_step, "q": q[:60], "dataset": dataset,
                        "gt": gt, "pred1": p1["strict"], "pred2": p2["strict"],
                    }
                    if ok2:
                        datum = _build_retry_datum(
                            q, dataset, solve_text, retry_text,
                            tokenizer, max_seq_len, types,
                        )
                        step_datums.append(datum)
                        total_trained += 1
                        log_entry["outcome"] = "retry_correct"
                        print(f"  [rt] ✓  retry correct  | step={global_step} | {q[:50]!r}")
                    else:
                        total_skipped += 1
                        log_entry["outcome"] = "retry_wrong"
                        print(f"  [rt] ✗  retry wrong    | step={global_step} | {q[:50]!r}")
                    step_log.append(log_entry)
                except Exception as exc:
                    print(f"  [baseline_rlvr][error] retry stage: {exc}")

        # ── Gradient update ───────────────────────────────────────────────────
        if step_datums:
            try:
                fb_future    = training_client.forward_backward(step_datums, "cross_entropy")
                optim_future = training_client.optim_step(adam_params)
                fb_result    = fb_future.result()
                optim_future.result()

                loss         = fb_result.metrics.get("loss:sum", float("nan"))
                global_step += 1
                hit_rate     = total_trained / max(total_rollouts, 1)
                print(
                    f"[baseline_rlvr] step {global_step:>4}/{max_steps}"
                    f"  loss={loss:.4f}"
                    f"  datums={len(step_datums)}"
                    f"  hit_rate={hit_rate:.1%}"
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
                print(f"[baseline_rlvr][error] optim step failed: {exc}")
        else:
            global_step += 1
            print(
                f"[baseline_rlvr] step {global_step:>4}/{max_steps}"
                f"  no datums  (skipped={total_skipped})"
            )

    # ── Save final checkpoint ─────────────────────────────────────────────────
    print(f"\n[baseline_rlvr] training complete.")
    print(f"[baseline_rlvr] rollouts={total_rollouts}"
          f"  trained={total_trained}  skipped={total_skipped}")
    print(f"[baseline_rlvr] hit rate: {total_trained / max(total_rollouts, 1):.2%}")
    _save_checkpoint(training_client, run_name, global_step=None, output_dir=output_dir)
    print(f"[baseline_rlvr] done.  Evaluate with:\n"
          f"        python scripts/eval_sft.py --run_name {run_name} --mode baseline_solve --dataset both")
