import json
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.infile, "r", encoding="utf-8") as fin, open(args.outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            ex["dataset"] = args.dataset
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1

    print(f"Patched {n} lines -> {args.outfile}")

if __name__ == "__main__":
    main()
