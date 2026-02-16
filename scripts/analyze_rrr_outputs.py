import json
import sys
from collections import defaultdict

# -------------------------
# Load jsonl
# -------------------------
def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# -------------------------
# accuracy helpers
# -------------------------
def first_correct(r):
    return bool(r["first"]["correct_loose"])


def system_correct(r):
    if r["first"]["correct_loose"]:
        return True
    if r.get("retry") and r["retry"]["correct_loose"]:
        return True
    return False


def avg_tokens(rows):
    tot = 0
    n = 0
    for r in rows:
        tot += r["first"]["meta"].get("total_tokens", 0)

        if r.get("reflection"):
            tot += r["reflection"]["meta"].get("total_tokens", 0)

        if r.get("retry"):
            tot += r["retry"]["meta"].get("total_tokens", 0)

        n += 1
    return tot / max(n, 1)


# -------------------------
# group by level
# -------------------------
def by_level(rows):
    buckets = defaultdict(list)
    for r in rows:
        lvl = (r.get("meta") or {}).get("level", "unknown")
        buckets[lvl].append(r)
    return buckets


# -------------------------
# accuracy
# -------------------------
def accuracy(rows, fn):
    vals = [fn(r) for r in rows]
    return sum(vals) / max(len(vals), 1)


# -------------------------
# main analysis
# -------------------------
def main(base_path, refl_path):

    base = load(base_path)
    refl = load(refl_path)

    print("\n===== OVERALL =====")

    base_first = accuracy(base, first_correct)
    refl_first = accuracy(refl, first_correct)
    refl_sys = accuracy(refl, system_correct)

    base_tok = avg_tokens(base)
    refl_tok = avg_tokens(refl)

    print(f"Baseline first accuracy : {base_first:.3f}")
    print(f"Reflect first accuracy  : {refl_first:.3f}")
    print(f"Reflect system accuracy : {refl_sys:.3f}")
    print(f"Δ accuracy (vs baseline): {refl_sys - base_first:+.3f}")
    print()

    print(f"Baseline avg tokens : {base_tok:.1f}")
    print(f"Reflect avg tokens  : {refl_tok:.1f}")
    print(f"Δ tokens            : {refl_tok - base_tok:+.1f}")
    print()

    if refl_tok > base_tok:
        gain_per_1k = (refl_sys - base_first) / (refl_tok - base_tok) * 1000
        print(f"Accuracy gain per 1k tokens: {gain_per_1k:.3f}")
    print()

    # ---------------------
    # level breakdown
    # ---------------------
    print("===== BY LEVEL =====")

    b_lv = by_level(base)
    r_lv = by_level(refl)

    levels = sorted(set(b_lv.keys()) | set(r_lv.keys()))

    print(f"{'Level':<8}{'N':<6}{'Base':<10}{'ReflSys':<10}{'Gain':<10}")

    for lvl in levels:
        b = b_lv.get(lvl, [])
        r = r_lv.get(lvl, [])
        if not b or not r:
            continue

        base_acc = accuracy(b, first_correct)
        refl_acc = accuracy(r, system_correct)
        gain = refl_acc - base_acc

        print(f"{str(lvl):<8}{len(b):<6}{base_acc:<10.3f}{refl_acc:<10.3f}{gain:<10.3f}")


# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python scripts/analyze_rrr_outputs.py baseline.jsonl reflect.jsonl")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
