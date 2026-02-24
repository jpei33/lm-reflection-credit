import os, json, argparse, asyncio
import tinker
from tinker import types

def load_conversations(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)              # e.g. Qwen/Qwen3-4B-Instruct-2507 (if available to Tinker)
    ap.add_argument("--train_jsonl", required=True)             # conversations.jsonl from Script A
    ap.add_argument("--project", default="rrr-sft")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--save_name", default="rrr_adapter")
    args = ap.parse_args()

    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("Missing TINKER_API_KEY in env")

    service = tinker.ServiceClient()

    # Create a LoRA training client (cookbook pattern) :contentReference[oaicite:2]{index=2}
    training_client = await service.create_lora_training_client_async(
        base_model=args.base_model,
        project=args.project,
        rank=args.rank,
        lr=args.lr,
    )

    # NOTE: The exact “rendering to Datum” step depends on the renderer/tokenization utilities.
    # The cookbook uses a renderer + Datum(model_input=..., loss_fn_inputs=...) before forward_backward. :contentReference[oaicite:3]{index=3}
    # Start by using the cookbook’s example format (messages JSONL). :contentReference[oaicite:4]{index=4}
    #
    # If you want, we can wire in a concrete renderer for your model’s chat format next.

    it = load_conversations(args.train_jsonl)

    for step in range(args.steps):
        batch = []
        for _ in range(args.batch_size):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        if not batch:
            break

        # TODO: convert `batch` (messages) -> list[types.Datum] (tokenized + masked targets)
        # datum_list = render_batch_to_datum_list(batch, args.base_model)

        # The cookbook then does forward/backward + optimizer step :contentReference[oaicite:5]{index=5}
        # await training_client.forward_backward_async(datum_list, loss_fn="cross_entropy")
        # await training_client.optim_step_async()

        if step % 10 == 0:
            print(f"step={step} (batch_size={len(batch)})")

    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=args.save_name)  # :contentReference[oaicite:6]{index=6}
    print("Saved adapter and created sampling client:", sampling_client)

if __name__ == "__main__":
    asyncio.run(main())
