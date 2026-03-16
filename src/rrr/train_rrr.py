"""
src/rrr/train_rrr.py
====================
RRR Training: Outcome-Gated Reflection Reward
---------------------------------------------
Online rejection-sampling fine-tuning (RFT) where:

  1. Current policy generates:  solve -> reflect -> retry
  2. Verifier checks whether the retry answer is correct
  3. Gradient is applied ONLY on successful trajectories
  4. Loss mask:  reflect tokens = 1.0,  retry tokens = 1.0
                 solve tokens   = 0.0  (context, not a training target)

Why this works
--------------
The model only receives a learning signal when reflection actually helped
the retry succeed.  This is "outcome-gated reflection reward": reflection
tokens are reinforced only when they demonstrably improved the answer.

Sampler refresh
---------------
Tinker separates training weights (training_client) from inference weights
(sampling_client).  We periodically call save_weights_and_get_sampling_client()
to push updated weights into the sampler, keeping rollouts roughly on-policy.

Entry point: scripts/train_rrr.py
"""

from __future__ import annotations

import json
import os
import random
import textwrap
from pathlib import Path
from typing import List, Optional

from src.rrr.rrr_infer import (
    build_reflection_prompt,
    build_retry_prompt,
    build_retry_prompt_grounded,
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

def _messages_to_datum(messages, tokenizer, max_seq_len: int, types,
                       completion_weight: float = 1.0):
    """
    Build a Tinker cross-entropy Datum that trains on the LAST assistant turn.
    All earlier turns are context (weight 0).
    completion_weight scales the loss on the target tokens (default 1.0).
    Identical logic to train_sft_lora_tiny.messages_to_datum.
    """
    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if messages[i]["role"] == "assistant"),
        None,
    )
    if last_asst is None:
        raise ValueError("No assistant message found in conversation.")

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
        raise ValueError("Empty completion after tokenisation – skipping.")

    max_completion = max_seq_len // 2
    max_prompt     = max_seq_len - min(len(completion_ids), max_completion)
    prompt_ids     = prompt_ids[-max_prompt:]
    completion_ids = completion_ids[:max_completion]

    P, C         = len(prompt_ids), len(completion_ids)
    full         = prompt_ids + completion_ids
    N            = P + C - 1
    raw_weights  = [0.0] * P + [completion_weight] * C
    seq_weights  = raw_weights[1:]          # left-shift to align with targets

    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=full[1:],    dtype="int64",   shape=[N]),
            "weights":       types.TensorData(data=seq_weights, dtype="float32", shape=[N]),
        },
    )


def _build_rrr_datums(
    question: str,
    dataset: str,
    solve_text: str,
    reflect_text: str,
    retry_text: str,
    reflection_mode: str,
    tokenizer,
    max_seq_len: int,
    types,
    solve_token_weight: float = 0.0,
) -> List:
    """
    Build Tinker Datums from one successful RRR trajectory.

    Datum 1 – Reflection turn
      context : [user: problem][asst: solve][user: reflect_request]
      target  : [asst: reflect]          ← weight 1.0

    Datum 2 – Retry turn
      context : [user: problem][asst: solve][user: reflect_request]
                [asst: reflect][user: retry_request]
      target  : [asst: retry]            ← weight 1.0

    Datum 0 (optional) – Solve turn, added when solve_token_weight > 0
      context : [user: problem]
      target  : [asst: solve]            ← weight solve_token_weight
      Giving the first-pass CoT a gradient signal means the model learns
      not just to reflect and retry well, but to avoid the errors it
      identifies — improving first-pass accuracy over training.
      A weight < 1.0 (e.g. 0.3) is recommended because the solve was wrong;
      we want a soft signal, not equal reinforcement of a bad solution.
    """
    solve_prompt_str  = build_solve_prompt(question, dataset)
    refl_request_str  = build_reflection_prompt(
        reflection_mode, question, solve_text, pred_final=None
    )
    retry_request_str = build_retry_prompt(question, reflect_text, dataset)

    msgs_reflect = [
        {"role": "user",      "content": solve_prompt_str},
        {"role": "assistant", "content": solve_text},
        {"role": "user",      "content": refl_request_str},
        {"role": "assistant", "content": reflect_text},
    ]
    msgs_retry = [
        {"role": "user",      "content": solve_prompt_str},
        {"role": "assistant", "content": solve_text},
        {"role": "user",      "content": refl_request_str},
        {"role": "assistant", "content": reflect_text},
        {"role": "user",      "content": retry_request_str},
        {"role": "assistant", "content": retry_text},
    ]

    datums = []

    # Datum 0: first-pass solve (optional, controlled by solve_token_weight)
    if solve_token_weight > 0.0:
        msgs_solve = [
            {"role": "user",      "content": solve_prompt_str},
            {"role": "assistant", "content": solve_text},
        ]
        try:
            datums.append(_messages_to_datum(
                msgs_solve, tokenizer, max_seq_len, types,
                completion_weight=solve_token_weight,
            ))
        except Exception as exc:
            print(f"  [rrr][warn] solve datum build skipped: {exc}")

    for msgs in (msgs_reflect, msgs_retry):
        try:
            datums.append(_messages_to_datum(msgs, tokenizer, max_seq_len, types))
        except Exception as exc:
            print(f"  [rrr][warn] datum build skipped: {exc}")
    return datums


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
            print(f"[rrr][warn] training data not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"[rrr] loaded {len(rows)} problems from {len(paths)} file(s).")
    return rows


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_rrr(
    # ── Tinker / model ───────────────────────────────────────────────────────
    base_model: str           = "Qwen/Qwen3-4B-Instruct-2507",
    sft_checkpoint: str       = None,       # run_name of the SFT warm-start
    run_name: str             = None,       # name to save this RRR checkpoint
    rank: int                 = 8,
    seed: int                 = 42,
    # ── Data ─────────────────────────────────────────────────────────────────
    train_jsonl_paths: List[str] = None,    # raw-problem JSONL files
    data_limit: Optional[int]    = None,    # cap total problems (smoke-test)
    # ── RL loop ──────────────────────────────────────────────────────────────
    max_steps: int            = 200,
    problems_per_step: int    = 4,          # problems sampled per gradient step
    sampler_refresh_every: int = 10,        # push weights to sampler every N steps
    # ── Reflection mode ──────────────────────────────────────────────────────
    reflection_mode: str      = "plan",     # "plan" | "full" | "tail"
    solve_token_weight: float = 0.0,        # >0 gives first-pass CoT a gradient signal
    # ── Decoding ─────────────────────────────────────────────────────────────
    solve_max_tokens: int     = 512,
    reflect_max_tokens: int   = 192,
    retry_max_tokens: int     = 512,
    temperature: float        = 0.7,
    top_p: float              = 0.95,
    reflect_temperature: float = 0.4,
    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float                 = 1e-5,
    beta1: float              = 0.9,
    beta2: float              = 0.95,
    weight_decay: float       = 0.0,
    grad_clip: float          = 1.0,
    max_seq_len: int          = 1024,
    # ── Checkpointing / resume ─────────────────────────────────────────────────
    checkpoint_every: int     = 0,
    resume: bool              = False,
    resume_step: int          = -1,
    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str           = "results/runs",
) -> None:

    # ── Resolve defaults ─────────────────────────────────────────────────────
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if train_jsonl_paths is None:
        train_jsonl_paths = [
            str(_REPO_ROOT / "data" / "processed" / "gsm8k_train.jsonl"),
            str(_REPO_ROOT / "data" / "processed" / "math_train.jsonl"),
        ]
    if run_name is None:
        run_name = f"rrr-{reflection_mode}-r{rank}-seed{seed}"

    print(textwrap.dedent(f"""
    +------------------------------------------------------+
    |       RRR Training  (Outcome-Gated Reflection)       |
    +------------------------------------------------------+
    |  base_model       : {base_model:<32s}|
    |  sft_checkpoint   : {(sft_checkpoint or 'none (base weights)'):<32s}|
    |  run_name         : {run_name:<32s}|
    |  reflection_mode  : {reflection_mode:<32s}|
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
        print(f"[rrr] resuming from step {start_step}: {tinker_uri}")
        training_client = service.create_training_client_from_state(
            path=tinker_uri,
            user_metadata={"run_name": run_name, "resumed_from_step": str(start_step)},
        )
        print(f"[rrr] training state restored (step {start_step}).")
    else:
        start_step = 0
        print("[rrr] creating LoRA training client ...")
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
            print(f"[rrr] loading SFT checkpoint '{sft_checkpoint}' from {_tinker_uri} ...")
            training_client.load_state(_tinker_uri).result()
            print("[rrr] SFT weights loaded.")
        else:
            print("[rrr] no SFT checkpoint – starting from base model weights.")

    tokenizer = training_client.get_tokenizer()
    print(f"[rrr] tokenizer ready  "
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
    print(f"[rrr] creating initial sampling client '{sampler_name}' ...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=sampler_name
    )
    print("[rrr] sampling client ready.\n")

    # ── Load training problems ────────────────────────────────────────────────
    problems = _load_problems(train_jsonl_paths, limit=data_limit)
    if not problems:
        raise RuntimeError("No training problems loaded. Check train_jsonl_paths.")
    rng = random.Random(seed)
    rng.shuffle(problems)

    # ── Counters ──────────────────────────────────────────────────────────────
    global_step     = start_step   # 0 for fresh runs, >0 when resuming
    total_rollouts  = 0
    total_successes = 0   # retry was correct  → gradient applied
    total_skipped   = 0   # first-try correct  → no RRR signal
    total_failed    = 0   # retry also wrong   → dropped
    problem_idx     = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"{run_name}_train_log.jsonl"

    # ── Main training loop ────────────────────────────────────────────────────
    while global_step < max_steps:

        # ── Refresh sampler to keep rollouts on-policy ───────────────────────
        if global_step > 0 and global_step % sampler_refresh_every == 0:
            print(f"[rrr][step {global_step}] refreshing sampling client ...")
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=sampler_name
            )

        # ── Collect datums for this gradient step ────────────────────────────
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

        # Collect solve results; keep only wrong ones for reflect/retry
        wrong_batch = []
        for (ex, q, dataset, gt, seed_base), fut in zip(step_examples, solve_futures):
            try:
                solve_text = _decode_future(fut, tokenizer)
                p1  = parse_pred_final(solve_text)
                ok1 = strict_match(ex, p1["strict"], gt)
                if ok1:
                    total_skipped += 1
                    step_log.append({"outcome": "skip_correct", "q": q[:60]})
                else:
                    wrong_batch.append((ex, q, dataset, gt, seed_base, solve_text, p1))
            except Exception as exc:
                print(f"  [rrr][error] solve stage: {exc}")

        # ── Stage 2: Fire all reflect requests in parallel ────────────────────
        reflect_futures = [
            _fire_sample(
                sampling_client, tokenizer,
                build_reflection_prompt(reflection_mode, q, solve_text, p1["loose"]),
                types, tinker,
                max_new_tokens=reflect_max_tokens,
                temperature=reflect_temperature, top_p=top_p,
                seed=seed_base + 1_000,
            )
            for (ex, q, dataset, gt, seed_base, solve_text, p1) in wrong_batch
        ]

        # Collect reflect results
        reflect_batch = []
        for (ex, q, dataset, gt, seed_base, solve_text, p1), fut in zip(wrong_batch, reflect_futures):
            try:
                reflect_text = _decode_future(fut, tokenizer)
                # Clamp to first 3 lines for old-style prompts (matches RRR eval behaviour).
                # Grounded reflections are also 3 lines but formatted differently — still safe to clamp.
                reflect_text = "\n".join(reflect_text.splitlines()[:3]).strip()
                reflect_batch.append((ex, q, dataset, gt, seed_base, solve_text, p1, reflect_text))
            except Exception as exc:
                print(f"  [rrr][error] reflect stage: {exc}")

        # ── Stage 3: Fire all retry requests in parallel ──────────────────────
        retry_futures = [
            _fire_sample(
                sampling_client, tokenizer,
                (build_retry_prompt_grounded(q, reflect_text, dataset)
                 if reflection_mode == "grounded"
                 else build_retry_prompt(q, reflect_text, dataset)),
                types, tinker,
                max_new_tokens=retry_max_tokens,
                temperature=temperature, top_p=top_p,
                seed=seed_base + 2_000,
            )
            for (ex, q, dataset, gt, seed_base, solve_text, p1, reflect_text) in reflect_batch
        ]

        # Collect retry results and build datums
        for (ex, q, dataset, gt, seed_base, solve_text, p1, reflect_text), fut in zip(reflect_batch, retry_futures):
            try:
                retry_text = _decode_future(fut, tokenizer)
                p2  = parse_pred_final(retry_text)
                ok2 = strict_match(ex, p2["strict"], gt)
                log_entry = {
                    "step": global_step, "q": q[:60], "dataset": dataset,
                    "gt": gt, "pred1": p1["strict"], "pred2": p2["strict"],
                }
                if ok2:
                    datums = _build_rrr_datums(
                        question=q, dataset=dataset,
                        solve_text=solve_text, reflect_text=reflect_text,
                        retry_text=retry_text, reflection_mode=reflection_mode,
                        tokenizer=tokenizer, max_seq_len=max_seq_len, types=types,
                        solve_token_weight=solve_token_weight,
                    )
                    step_datums.extend(datums)
                    total_successes += 1
                    log_entry["outcome"] = "success"
                    print(f"  [rrr] ✓  retry correct  | step={global_step} | {q[:50]!r}")
                else:
                    total_failed += 1
                    log_entry["outcome"] = "fail"
                    print(f"  [rrr] ✗  retry wrong    | step={global_step} | {q[:50]!r}")
                step_log.append(log_entry)
            except Exception as exc:
                print(f"  [rrr][error] retry stage: {exc}")

        # ── Gradient update ───────────────────────────────────────────────────
        if step_datums:
            try:
                fb_future    = training_client.forward_backward(step_datums, "cross_entropy")
                optim_future = training_client.optim_step(adam_params)
                fb_result    = fb_future.result()
                optim_future.result()

                loss         = fb_result.metrics.get("loss:sum", float("nan"))
                global_step += 1
                rate         = total_successes / max(total_rollouts, 1)
                print(
                    f"[rrr] step {global_step:>4}/{max_steps}"
                    f"  loss={loss:.4f}"
                    f"  datums={len(step_datums)}"
                    f"  success_rate={rate:.1%}"
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
                print(f"[rrr][error] optim step failed at global_step={global_step}: {exc}")
        else:
            global_step += 1
            print(
                f"[rrr] step {global_step:>4}/{max_steps}"
                f"  no datums this step  "
                f"(skipped={total_skipped}  failed={total_failed})"
            )

    # ── Save final checkpoint ─────────────────────────────────────────────────
    print(f"\n[rrr] training complete.")
    print(f"[rrr] rollouts={total_rollouts}  successes={total_successes}"
          f"  skipped={total_skipped}  failed={total_failed}")
    print(f"[rrr] overall success rate: {total_successes / max(total_rollouts, 1):.2%}")
    _save_checkpoint(training_client, run_name, global_step=None, output_dir=output_dir)
    print(f"[rrr] done.  Evaluate with:\n"
          f"       python scripts/eval_sft.py --run_name {run_name} --mode reflect_{reflection_mode}_retry --dataset both")
