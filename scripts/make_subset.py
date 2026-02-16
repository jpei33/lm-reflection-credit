import json, argparse, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    rng = random.Random(args.seed)
    rng.shuffle(data)
    data = data[:args.n]

    with open(args.outfile, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} examples to {args.outfile}")

if __name__ == "__main__":
    main()
