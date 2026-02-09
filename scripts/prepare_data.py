from datasets import load_dataset
import json
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    ds = load_dataset("gsm8k", "main")
    train = ds["train"]

    out_file = OUT / "gsm8k_train.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in train:
            record = {
                "question": ex["question"],
                "answer": ex["answer"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(train)} examples to {out_file}")

if __name__ == "__main__":
    main()
