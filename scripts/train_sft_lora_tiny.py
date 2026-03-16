"""
train_sft_lora_tiny.py
----------------------
SFT training script using the Tinker API (Thinking Machines Lab).
Trains a LoRA adapter with full reproducibility and curriculum support.

Follows the Tinker low-level cookbook pattern:
  forward_backward -> (xgrad_accum) -> optim_step -> ... -> save_weights_and_get_sampling_client

Seeds are set in three places for publication-grade reproducibility:
  1. Data shuffle   - seeded Python RNG (local)
  2. LoRA init      - `seed` param in create_lora_training_client (remote worker)
  3. Sampling       - `seed` field in SamplingParams (remote worker)

--- Running the 4 experimental conditions (fair comparison) ---
All 4 runs use --max_steps 500 so every model trains for the same
number of optimizer steps regardless of dataset size.

  python scripts\\train_sft_lora_tiny.py --mode baseline_solve     --max_steps 500 --seed 42
  python scripts\\train_sft_lora_tiny.py --mode retry_only         --max_steps 500 --seed 42
  python scripts\\train_sft_lora_tiny.py --mode reflect_full_retry --max_steps 500 --seed 42
  python scripts\\train_sft_lora_tiny.py --mode reflect_plan_retry --max_steps 500 --seed 42

--mode auto-sets --train_jsonl to data/processed/curriculum/{mode}.jsonl
and includes the mode name in --run_name for easy identification.
Add --curriculum to preserve the easy->hard sort order of the dataset.

Requirements:
    pip install tinker transformers torch numpy python-dotenv tqdm
"""

import os
import json
import random
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from tqdm import tqdm

# Resolve .env from the repo root (one level up from scripts/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# -----------------------------------------------------------------------------
# Seed utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python's RNG for deterministic data shuffling."""
    random.seed(seed)
    print(f"[seed] local RNG seeded with {seed}")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_sft_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load a JSONL where each line is {"messages": [...], ...}."""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def shuffle_data(rows: List[Dict], seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows


# -----------------------------------------------------------------------------
# Chat-template rendering + SFT loss masking -> Tinker Datum
# -----------------------------------------------------------------------------

def _has_chat_template(tokenizer) -> bool:
    """Check whether a tokenizer has a usable chat template."""
    tmpl = getattr(tokenizer, "chat_template", None)
    return tmpl is not None


def _to_ids(result) -> List[int]:
    """
    Normalize any tokenizer output to a plain List[int].

    HuggingFace fast tokenizers can return several types depending on the
    method and version:
      - List[int]          - already correct
      - BatchEncoding      - dict-like with an "input_ids" key
      - tokenizers.Encoding - Rust-backend object with an .ids attribute
      - torch.Tensor / np.ndarray - from return_tensors="pt"/"np"

    EncodedTextChunk (Pydantic strict) requires a plain list of Python ints,
    so we normalise everything here before passing to Tinker.
    """
    # BatchEncoding or plain dict: {"input_ids": [...]}
    if hasattr(result, "input_ids"):
        result = result.input_ids
    elif isinstance(result, dict) and "input_ids" in result:
        result = result["input_ids"]

    # tokenizers.Encoding (Rust fast tokenizer backend)
    if hasattr(result, "ids"):
        return [int(x) for x in result.ids]

    # Tensor (torch / numpy)
    if hasattr(result, "tolist"):
        return [int(x) for x in result.tolist()]

    # Already a sequence - cast each element to int to be safe
    return [int(x) for x in result]


def _format_messages_plain(messages: List[Dict[str, str]]) -> str:
    """
    Fallback formatter when the tokenizer has no chat_template.
    Uses a simple role-header format that works with any base model.
    Returns (full_text, list_of_assistant_spans) where spans are (start_char, end_char).
    """
    parts = []
    for m in messages:
        role = m["role"].capitalize()
        parts.append(f"### {role}\n{m['content'].strip()}")
    return "\n\n".join(parts) + "\n"


def messages_to_datum(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
    types,
) -> object:
    """
    Tokenize a conversation and build a Tinker Datum for SFT.

    Tinker's cross_entropy loss uses a causal next-token prediction setup:
      - model_input    - full_ids[:-1]: the full context (prompt + completion)
                         minus the last token, so the model sees history.
      - target_tokens  - full_ids[1:]: the next-token targets (left-shifted).
      - weights        - float array aligned with target_tokens: 0.0 for
                         positions corresponding to prompt tokens (no loss),
                         1.0 for completion token positions.

    Both target_tokens and weights are required by the cross_entropy loss.

    For multi-turn conversations we train on the LAST assistant turn only,
    giving the model full conversation context as the prompt.
    """
    # Find the last assistant turn
    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
        None,
    )
    if last_asst is None:
        raise ValueError("No assistant message in conversation - skipping.")

    prompt_msgs = messages[:last_asst]   # everything before the last assistant turn

    if _has_chat_template(tokenizer):
        # -- Chat-template path ---------------------------------------------
        # Prompt: all turns up to (not including) the last assistant turn,
        # with add_generation_prompt=True so the assistant header is appended.
        prompt_ids: List[int] = _to_ids(tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        ))
        # Full sequence including the last assistant turn
        full_ids: List[int] = _to_ids(tokenizer.apply_chat_template(
            messages[:last_asst + 1],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        ))
        # Completion = tokens added by the last assistant turn
        completion_ids: List[int] = full_ids[len(prompt_ids):]

    else:
        # -- Plain-text fallback --------------------------------------------
        prompt_parts = [
            f"### {m['role'].capitalize()}\n{m['content'].strip()}"
            for m in prompt_msgs
        ]
        prompt_text = "\n\n".join(prompt_parts) + ("\n\n" if prompt_parts else "") + "### Assistant\n"
        prompt_ids = _to_ids(tokenizer.encode(prompt_text, add_special_tokens=True))
        completion_ids = _to_ids(tokenizer.encode(
            messages[last_asst]["content"].strip(),
            add_special_tokens=False,
        ))

    if not completion_ids:
        raise ValueError("Empty completion after tokenization - skipping.")

    # Truncate: reserve half of max_seq_len for completion, rest for prompt
    max_completion = max_seq_len // 2
    max_prompt     = max_seq_len - min(len(completion_ids), max_completion)
    prompt_ids     = prompt_ids[-max_prompt:]        # keep the tail (most recent context)
    completion_ids = completion_ids[:max_completion]

    # Build Datum using Tinker's causal-LM SFT pattern:
    #
    #   full_ids       = prompt_ids + completion_ids  (the complete token sequence)
    #   model_input    = full_ids[:-1]  (all but the last token - right-shifted context)
    #   target_tokens  = full_ids[1:]   (all but the first token - left-shifted next-token targets)
    #   weights        = [0.0]*len(prompt_ids) + [1.0]*len(completion_ids)
    #                    then shifted left by 1 to align with target_tokens.
    #                    Prompt positions get weight 0 (no loss), completion gets 1.
    #
    # Tinker's cross_entropy loss requires BOTH target_tokens and weights.
    # Both TensorData objects must include shape so the server can convert
    # them to array records.
    P = len(prompt_ids)
    C = len(completion_ids)
    full_ids  = prompt_ids + completion_ids         # length P+C
    seq_input = full_ids[:-1]                       # length P+C-1  (model context)
    seq_target = full_ids[1:]                       # length P+C-1  (next-token targets)
    raw_weights = [0.0] * P + [1.0] * C            # length P+C
    seq_weights = raw_weights[1:]                   # length P+C-1  (aligned with targets)

    N = len(seq_target)  # == P+C-1
    target_tensor = types.TensorData(data=seq_target, dtype="int64",   shape=[N])
    weight_tensor = types.TensorData(data=seq_weights, dtype="float32", shape=[N])
    return types.Datum(
        model_input=types.ModelInput.from_ints(seq_input),
        loss_fn_inputs={
            "target_tokens": target_tensor,
            "weights":        weight_tensor,
        },
    )


def build_datums(
    batch: List[Dict],
    tokenizer,
    max_seq_len: int,
    types,
) -> list:
    datums = []
    for row in batch:
        try:
            d = messages_to_datum(row["messages"], tokenizer, max_seq_len, types)
            datums.append(d)
        except Exception as exc:
            import traceback
            print(f"  [warn] skipped example ({type(exc).__name__}): {exc}")
            traceback.print_exc()
    return datums


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def batched(rows: List[Dict], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tiny SFT training via Tinker LoRA API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model --
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="Tinker base model identifier")

    # -- Experimental condition --
    ap.add_argument("--mode", default=None,
                    choices=["baseline_solve", "retry_only",
                             "reflect_full_retry", "reflect_plan_retry",
                             "reflect_grounded"],
                    help=(
                        "Training condition. Auto-sets --train_jsonl to "
                        "data/processed/curriculum/{mode}.jsonl and includes "
                        "the mode name in --run_name. "
                        "'reflect_grounded' uses teacher-generated forced-quote reflections."
                    ))

    # -- Data --
    ap.add_argument("--train_jsonl", default=None,
                    help="Path to training JSONL. Defaults to the curriculum "
                         "file for --mode, or data/processed/tiny_sft_500.jsonl "
                         "if --mode is not set.")
    ap.add_argument("--curriculum", action="store_true", default=False,
                    help="Preserve dataset sort order (no shuffle). "
                         "Use with curriculum-ordered files so easy examples "
                         "come first.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples for a quick smoke test")

    # -- LoRA --
    lora = ap.add_argument_group("LoRA config")
    lora.add_argument("--rank", type=int, default=8,
                      help="LoRA rank (size of low-rank matrices)")
    lora.add_argument("--train_attn",   action="store_true", default=True,
                      help="Add LoRA to attention layers")
    lora.add_argument("--train_mlp",    action="store_true", default=True,
                      help="Add LoRA to MLP layers")
    lora.add_argument("--train_unembed",action="store_true", default=True,
                      help="Add LoRA to unembedding layer")

    # -- Optimiser --
    opt = ap.add_argument_group("Optimiser (AdamW)")
    opt.add_argument("--lr",           type=float, default=2e-4)
    opt.add_argument("--beta1",        type=float, default=0.9)
    opt.add_argument("--beta2",        type=float, default=0.95)
    opt.add_argument("--weight_decay", type=float, default=0.0)
    opt.add_argument("--grad_clip",    type=float, default=1.0,
                     help="Global gradient norm clip (0 = off)")
    opt.add_argument("--epochs",       type=int,   default=1)
    opt.add_argument("--max_steps",    type=int,   default=0,
                     help="Stop after this many optimizer steps regardless of "
                          "epoch/dataset size. Use to match training budget "
                          "fairly across the 4 conditions (0 = train full dataset).")
    opt.add_argument("--batch_size",   type=int,   default=4,
                     help="Examples per forward_backward call")
    opt.add_argument("--grad_accum",   type=int,   default=4,
                     help="Micro-batches before optim_step (effective batch = batch_size x grad_accum)")
    opt.add_argument("--max_seq_len",  type=int,   default=1024)

    # -- Checkpoint / resume --
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="Call save_state every N optimizer steps (0 = off)")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume training from the latest (or --resume_step) checkpoint "
                         "for this run_name, restoring weights and Adam state.")
    ap.add_argument("--resume_step", type=int, default=-1,
                    help="Step to resume from (-1 = auto-detect highest mid-run snapshot, "
                         "falling back to the final checkpoint).")

    # -- Sampling (post-training eval) --
    samp = ap.add_argument_group("Post-training sampling")
    samp.add_argument("--sample_prompt",  default="Solve: What is 7 * 8?")
    samp.add_argument("--max_new_tokens", type=int,   default=128)
    samp.add_argument("--temperature",    type=float, default=0.0,
                      help="0 = greedy (fully deterministic, seed has no effect)")

    # -- Reproducibility --
    rep = ap.add_argument_group("Reproducibility")
    rep.add_argument(
        "--seed", type=int, default=42,
        help=(
            "Master seed. Controls: "
            "(1) data shuffle order [local], "
            "(2) LoRA weight init via create_lora_training_client(seed=) [remote], "
            "(3) stochastic sampling via SamplingParams(seed=) [remote]."
        ),
    )

    # -- Run metadata --
    meta = ap.add_argument_group("Run metadata")
    meta.add_argument("--run_name", default=None,
                      help="Defaults to qwen3-4b-{mode}-r{rank}-seed{seed} "
                           "when --mode is set, else lora-r{rank}-seed{seed}")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Resume helper
# -----------------------------------------------------------------------------

def _resolve_resume(run_name: str, resume_step: int, output_dir: str):
    """
    Locate a checkpoint to resume SFT from; return (tinker_uri, start_step).

    Priority:
      1. --resume_step N  → {run_name}-step{N}.checkpoint.json
      2. auto             → highest {run_name}-step*.checkpoint.json
      3. fallback         → {run_name}.checkpoint.json (needs steps_completed field)
    """
    import json as _json
    import re as _re
    from pathlib import Path as _Path

    ckpt_dir = _Path(output_dir)

    if resume_step > 0:
        p = ckpt_dir / f"{run_name}-step{resume_step}.checkpoint.json"
        if not p.exists():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {p}")
        return _json.loads(p.read_text())["tinker_path"], resume_step

    pattern = _re.compile(rf"^{_re.escape(run_name)}-step(\d+)\.checkpoint\.json$")
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Resolve mode -> train_jsonl and run_name ---------------------------
    _CURRICULUM_DIR = _REPO_ROOT / "data" / "processed" / "curriculum"
    _FALLBACK_JSONL = _REPO_ROOT / "data" / "processed" / "tiny_sft_500.jsonl"

    if args.mode is not None and args.train_jsonl is None:
        args.train_jsonl = str(_CURRICULUM_DIR / f"{args.mode}.jsonl")
    elif args.train_jsonl is None:
        args.train_jsonl = str(_FALLBACK_JSONL)

    mode_tag = args.mode or "custom"
    if args.run_name is None:
        args.run_name = f"qwen3-4b-{mode_tag}-r{args.rank}-seed{args.seed}"

    max_steps_str = str(args.max_steps) if args.max_steps > 0 else "unlimited"
    print(textwrap.dedent(f"""
    +------------------------------------------------------+
    |         SFT - Tinker LoRA Training                   |
    +------------------------------------------------------+
    |  base_model   : {args.base_model:<36s}|
    |  mode         : {mode_tag:<36s}|
    |  train_jsonl  : {args.train_jsonl:<36s}|
    |  curriculum   : {str(args.curriculum):<36s}|
    |  LoRA rank    : {args.rank:<36d}|
    |  lr           : {args.lr:<36g}|
    |  epochs       : {args.epochs:<36d}|
    |  max_steps    : {max_steps_str:<36s}|
    |  batch_size   : {args.batch_size:<36d}|
    |  grad_accum   : {args.grad_accum:<36d}|
    |  max_seq_len  : {args.max_seq_len:<36d}|
    |  seed         : {args.seed:<36d}|
    |  run_name     : {args.run_name:<36s}|
    +------------------------------------------------------+
    """))

    # -- Local RNG seed (data shuffle) -------------------------------------
    set_seed(args.seed)

    # -- API key check ------------------------------------------------------
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TINKER_API_KEY not found. Add it to your .env file or export it."
        )

    # -- Load data ----------------------------------------------------------
    rows = load_sft_jsonl(args.train_jsonl, limit=args.limit)
    if args.curriculum:
        print(f"[data] {len(rows)} examples loaded (curriculum order preserved)")
    else:
        rows = shuffle_data(rows, seed=args.seed)
        print(f"[data] {len(rows)} examples loaded and shuffled (seed={args.seed})")

    # -- Tinker setup -------------------------------------------------------
    import tinker
    from tinker import types

    service = tinker.ServiceClient()

    if args.resume:
        # Restore weights + Adam state from a prior checkpoint.
        _resume_uri, _start_step = _resolve_resume(
            args.run_name, args.resume_step,
            str(_REPO_ROOT / args.output_dir if hasattr(args, "output_dir") else _REPO_ROOT / "results/runs")
        )
        print(f"[tinker] resuming from step {_start_step}: {_resume_uri}")
        training_client = service.create_training_client_from_state(
            path=_resume_uri,
            user_metadata={"run_name": args.run_name, "resumed_from_step": str(_start_step)},
        )
        print(f"[tinker] training state restored (step {_start_step}).")

        # Advance the data iterator to approximately the right position so we
        # don't re-train on examples already seen in the previous run.
        # Each optimizer step consumed grad_accum * batch_size examples.
        _examples_seen = _start_step * args.grad_accum * args.batch_size
        _skip = _examples_seen % len(rows)   # wrap within epoch
        if _skip > 0:
            rows = rows[_skip:] + rows[:_skip]
            print(f"[data] skipped {_skip} examples to resume at approx. step {_start_step}.")
    else:
        _start_step = 0
        # create_lora_training_client takes: base_model, rank, seed, train_mlp,
        # train_attn, train_unembed, user_metadata.
        # seed here controls LoRA weight initialisation on the remote GPU worker.
        print(f"[tinker] creating LoRA training client (rank={args.rank}, seed={args.seed}) ...")
        training_client = service.create_lora_training_client(
            base_model=args.base_model,
            rank=args.rank,
            seed=args.seed,              # <- reproducible LoRA init (remote)
            train_attn=args.train_attn,
            train_mlp=args.train_mlp,
            train_unembed=args.train_unembed,
            user_metadata={"run_name": args.run_name, "seed": str(args.seed)},
        )
    print("[tinker] training client ready.")

    # -- Load tokenizer via Tinker (correct tokenizer for the remote model) -
    # training_client.get_tokenizer() fetches the model's tokenizer from Tinker's
    # servers, applying any overrides (e.g. Llama uses the instruct tokenizer
    # which has a chat_template, unlike the raw base-model tokenizer on HF).
    print(f"[data] loading tokenizer via training_client.get_tokenizer() ...")
    tokenizer = training_client.get_tokenizer()
    has_tmpl = _has_chat_template(tokenizer)
    print(f"[data] tokenizer loaded - chat_template={'yes' if has_tmpl else 'no (plain-text fallback)'}")

    # Adam params - passed on every optim_step call
    adam_params = types.AdamParams(
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
    )

    # -- SFT training loop --------------------------------------------------
    # Pattern (from Tinker docs):
    #   fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
    #   optim_future  = training_client.optim_step(adam_params)      # fire immediately
    #   fwdbwd_future.result()                                        # then wait
    #   optim_future.result()
    #
    # With gradient accumulation we fire grad_accum forward_backward calls,
    # then a single optim_step, then await everything.

    global_step  = _start_step   # 0 for fresh runs; >0 when resuming
    micro_step   = 0     # forward_backward calls since last optim_step
    pending_fb   = []    # APIFutures for current accumulation window
    budget_hit   = False # set True when --max_steps is reached

    # Progress bar: if max_steps is set, show steps not batches
    if args.max_steps > 0:
        pbar = tqdm(total=args.max_steps, desc="training", unit="step")
    else:
        total_batches = args.epochs * max(1, len(rows) // args.batch_size)
        pbar = tqdm(total=total_batches, desc="training", unit="batch")

    for epoch in range(1, args.epochs + 1):
        if budget_hit:
            break
        for batch in batched(rows, args.batch_size):
            if budget_hit:
                break

            datums = build_datums(batch, tokenizer, args.max_seq_len, types)
            if not datums:
                if args.max_steps == 0:
                    pbar.update(1)
                continue

            # -- forward + backward (fire-and-don't-wait-yet) ---------------
            fb_future = training_client.forward_backward(datums, "cross_entropy")
            pending_fb.append(fb_future)
            micro_step += 1

            # -- optimizer step after grad_accum micro-batches --------------
            if micro_step % args.grad_accum == 0:
                optim_future = training_client.optim_step(adam_params)

                losses = []
                for f in pending_fb:
                    result = f.result()
                    losses.append(result.metrics.get("loss:sum", float("nan")))
                optim_future.result()

                avg_loss = sum(losses) / len(losses) if losses else float("nan")
                global_step += 1
                pending_fb = []

                pbar.set_postfix(
                    epoch=f"{epoch}/{args.epochs}",
                    step=global_step,
                    loss=f"{avg_loss:.4f}",
                )
                if args.max_steps > 0:
                    pbar.update(1)

                # Mid-run checkpoint
                if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    import json as _ckpt_json
                    from pathlib import Path as _ckpt_path
                    ckpt_name = f"{args.run_name}-step{global_step}"
                    _ckpt_resp = training_client.save_state(ckpt_name).result()
                    _ckpt_meta = _ckpt_path(_REPO_ROOT / "results" / "runs" / f"{ckpt_name}.checkpoint.json")
                    _ckpt_meta.write_text(_ckpt_json.dumps({
                        "run_name": ckpt_name,
                        "tinker_path": _ckpt_resp.path,
                        "steps_completed": global_step,
                    }, indent=2))
                    print(f"\n[ckpt] saved state: {ckpt_name} -> {_ckpt_resp.path}")

                # Step budget check
                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"\n[train] --max_steps {args.max_steps} reached - stopping.")
                    budget_hit = True
                    break

            if args.max_steps == 0:
                pbar.update(1)

        # -- flush remaining accumulated gradients at epoch boundary --------
        if pending_fb and not budget_hit:
            optim_future = training_client.optim_step(adam_params)
            losses = [f.result().metrics.get("loss:sum", float("nan")) for f in pending_fb]
            optim_future.result()
            global_step += 1
            pending_fb = []
            print(f"\n[flush] epoch {epoch} boundary - step {global_step}, "
                  f"loss={sum(losses)/len(losses):.4f}")

    pbar.close()
    print(f"\n[train] done - {global_step} optimizer steps over {args.epochs} epoch(s).")

    # -- Save LoRA adapter --------------------------------------------------
    # save_state() persists the checkpoint on Tinker's servers and returns a
    # Tinker URI (e.g. "tinker://run-id/weights/...") that eval_sft.py can
    # use to reload the exact weights via create_training_client_from_state().
    # We write the URI to a sidecar JSON file so eval_sft.py can find it.
    print(f"[save] persisting checkpoint via save_state('{args.run_name}') ...")
    save_resp = training_client.save_state(args.run_name).result()
    tinker_path = save_resp.path
    print(f"[save] checkpoint URI: {tinker_path}")

    # Write sidecar metadata file for eval_sft.py
    _ckpt_dir = _REPO_ROOT / "results" / "runs"
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    _meta_path = _ckpt_dir / f"{args.run_name}.checkpoint.json"
    import json as _json
    _meta_path.write_text(
        _json.dumps({
            "run_name": args.run_name,
            "tinker_path": tinker_path,
            "steps_completed": global_step,
        }, indent=2)
    )
    print(f"[save] checkpoint metadata written to: {_meta_path}")

    # Get a sampling client for the post-training sanity check below
    sampling_client = training_client.save_weights_and_get_sampling_client()
    print(f"[save] sampling client ready for sanity check.")

    # -- Post-training sanity sample ----------------------------------------
    # SamplingParams.seed sets the remote PRNG -> reproducible stochastic outputs.
    # temperature=0 is fully greedy, seed has no effect in that mode.
    print("\n[sample] running post-training sanity check ...")
    if _has_chat_template(tokenizer):
        prompt_tokens = _to_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": args.sample_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        ))
    else:
        prompt_tokens = _to_ids(tokenizer.encode(
            f"### User\n{args.sample_prompt}\n\n### Assistant\n",
            add_special_tokens=True,
        ))
    sample_future = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,              # <- reproducible sampling (remote)
        ),
    )
    sample_result = sample_future.result()
    response_text = tokenizer.decode(sample_result.sequences[0].tokens, skip_special_tokens=True)
    print(f"[sample] prompt  : {args.sample_prompt!r}")
    print(f"[sample] response: {response_text}")

    # -- Final summary ------------------------------------------------------
    print(textwrap.dedent(f"""
    +------------------------------------------------------+
    |                  Training Complete                   |
    +------------------------------------------------------+
    |  run_name     : {args.run_name:<36s}|
    |  base_model   : {args.base_model:<36s}|
    |  mode         : {mode_tag:<36s}|
    |  LoRA rank    : {args.rank:<36d}|
    |  lr           : {args.lr:<36g}|
    |  epochs       : {args.epochs:<36d}|
    |  max_steps    : {max_steps_str:<36s}|
    |  seed         : {args.seed:<36d}|
    |  n_examples   : {len(rows):<36d}|
    |  opt_steps    : {global_step:<36d}|
    +------------------------------------------------------+
    """))


if __name__ == "__main__":
    main()
