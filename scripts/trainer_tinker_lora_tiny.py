# scripts/train_tinker_lora_tiny.py
import os
import json
import argparse
from typing import List, Dict

def format_messages_as_text(messages: List[Dict[str, str]]) -> str:
    """
    Minimal chat serialization.
    Works for SFT where labels are assistant messages.
    """
    parts = []
    for m in messages:
        role = m["role"]
        content = (m.get("content") or "").strip()
        if role == "user":
            parts.append(f"### User\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant\n{content}")
        else:
            parts.append(f"### {role.title()}\n{content}")
    return "\n\n".join(parts).strip() + "\n"

def load_sft_jsonl(path: str, limit: int = None) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_tinker_sft_file(rows: List[Dict], out_path: str) -> None:
    """
    Writes a simple JSONL where each line is:
      {"text": "..."}
    Many trainers accept this minimal format.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            text = format_messages_as_text(r["messages"])
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--in_sft", default="data/processed/tiny_sft_500.jsonl")
    ap.add_argument("--work_dir", default="data/processed/tinker_tmp")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="rrr-tiny-sft")
    ap.add_argument("--run_name", default="tiny-500-1epoch-lora")
    args = ap.parse_args()

    # --- Prepare training file ---
    rows = load_sft_jsonl(args.in_sft)
    out_train = os.path.join(args.work_dir, "tiny_sft_500_text.jsonl")
    write_tinker_sft_file(rows, out_train)
    print(f"[prep] wrote tinker training jsonl: {out_train} ({len(rows)} rows)")

    # --- Tinker training ---
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TINKER_API_KEY in environment.")

    import tinker
    from tinker import types

    service = tinker.ServiceClient()

    # Create LoRA training client
    training_client = service.create_lora_training_client(
        project_name=args.project,
        base_model=args.base_model,
    )

    # Upload / register dataset (exact field names may differ slightly)
    dataset = training_client.create_dataset_from_jsonl(
        name=f"{args.run_name}-dataset",
        path=out_train,
        text_field="text",
    )

    # LoRA config (seed here is for training determinism, if supported)
    lora_cfg = types.LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        # If types.LoraConfig doesn't support seed, remove this line:
        seed=args.seed,
    )

    train_cfg = types.SftTrainConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        # If train config supports it, set seed here too:
        seed=args.seed,
    )

    job = training_client.train_sft(
        dataset_id=dataset.id,
        lora_config=lora_cfg,
        train_config=train_cfg,
        run_name=args.run_name,
    )

    print(f"[train] started job: {job.id}")

    # Wait until finished (blocking)
    job = training_client.wait_for_job(job.id)
    print(f"[train] finished job status: {job.status}")

    # Save weights and get a sampling client for inference/eval
    sampling_client = training_client.save_weights_and_get_sampling_client(job.id)
    print("[train] LoRA weights saved.")
    print(f"[train] sampling client model ref: {sampling_client.model_ref}")

if __name__ == "__main__":
    main()
