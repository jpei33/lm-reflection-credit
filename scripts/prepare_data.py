from datasets import load_dataset, concatenate_datasets
import json
from pathlib import Path
import argparse
import re

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_last_boxed_balanced(text: str):
    """
    Return the content of the LAST \\boxed{...} using balanced-brace parsing.
    Handles nested braces like \\boxed{\\frac{\\sqrt6}{3}}.
    """
    if not text:
        return None

    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    i = idx + len(r"\boxed{")
    depth = 1
    start = i

    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None  # unbalanced

    return text[start:i-1].strip()


def _extract_boxed_final(solution: str) -> str:
    """
    MATH solutions typically end with \\boxed{...}.
    Return the last boxed content if present; otherwise fall back to last line.
    """
    if not solution:
        return ""
    matches = _BOXED_RE.findall(solution)
    if matches:
        return matches[-1].strip()

    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    return lines[-1] if lines else solution.strip()

def dump_jsonl(records, out_file: Path):
    n = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} examples to {out_file}")

def prepare_gsm8k(split: str):
    ds = load_dataset("gsm8k", "main")[split]
    for i, ex in enumerate(ds):
        yield {
            "id": f"gsm8k_{split}_{i:06d}",
            "dataset": "gsm8k",
            "split": split,
            "question": ex["question"],
            "answer": ex["answer"],
            "meta": {}
        }

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

def prepare_math(split: str, keep_solution: bool = False):
    # Load each subject then concatenate into one dataset for the split
    dsets = [load_dataset("EleutherAI/hendrycks_math", subj)[split] for subj in MATH_SUBJECTS]
    ds = concatenate_datasets(dsets)

    for i, ex in enumerate(ds):
        problem = ex.get("problem", "")
        solution = ex.get("solution", "")
        answer = extract_last_boxed_balanced(solution)
        if not answer:
            # fallback: last non-empty line
            lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
            answer = lines[-1] if lines else ""

        meta = {
            "level": ex.get("level"),
            "type": ex.get("type"),
        }
        if keep_solution:
            meta["solution"] = solution  # optional, can be large

        yield {
            "id": ex.get("unique_id", f"math_{split}_{i:06d}"),
            "dataset": "math",
            "split": split,
            "question": str(problem).strip(),
            "answer": answer,
            "meta": meta,
        }

PREPARERS = {
    "gsm8k": prepare_gsm8k,
    "math": prepare_math,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=PREPARERS.keys(), required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep_solution", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset
    split = args.split
    out_file = Path(args.out) if args.out else (OUT / f"{dataset}_{split}.jsonl")

    if dataset == "math":
        records = PREPARERS[dataset](split, keep_solution=args.keep_solution)
    else:
        records = PREPARERS[dataset](split)

    dump_jsonl(records, out_file)

if __name__ == "__main__":
    main()
