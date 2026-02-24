import json
import argparse
from pathlib import Path

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrr_eval_jsonl", required=True)   # your /rrr_runs/*.jsonl
    ap.add_argument("--out_jsonl", required=True)        # conversations.jsonl
    ap.add_argument("--mode", required=True, choices=["baseline","retry_only","reflect_full","reflect_plan"])
    args = ap.parse_args()

    rows = []
    with open(args.rrr_eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") in ("skipped","failed"):
                continue
            q = rec.get("question","")
            dataset = (rec.get("dataset") or "").lower()

            # You’ll use the SAME prompt builders as rrr_infer.py
            # Import them if you want, or duplicate minimal versions here.

            if args.mode == "baseline":
                user = build_solve_prompt(q, dataset)
                # Train to match your dataset’s gold formatting:
                # Use rec["gt_final"] only if your target format is just final.
                # Better: store gold solutions in your dataset and use those.
                assistant = rec["first"]["solution"]  # or a gold solution string if you have it
                rows.append({"messages":[{"role":"user","content":user},{"role":"assistant","content":assistant}]})

            elif args.mode == "retry_only":
                user = build_retry_prompt(q, reflection="", dataset=dataset)
                assistant = rec["retry"]["solution"] if rec.get("retry") else rec["first"]["solution"]
                rows.append({"messages":[{"role":"user","content":user},{"role":"assistant","content":assistant}]})

            elif args.mode in ("reflect_full","reflect_plan"):
                refl = rec.get("reflection")
                retry = rec.get("retry")
                if not refl or not retry:
                    continue

                # reflection task
                if args.mode == "reflect_full":
                    user_refl = build_reflection_prompt_full(q, rec["first"]["solution"], rec["first"]["pred_final_loose"])
                else:
                    user_refl = build_reflection_prompt_plan(q, rec["first"]["pred_final_loose"])
                assistant_refl = refl["text"]
                rows.append({"messages":[{"role":"user","content":user_refl},{"role":"assistant","content":assistant_refl}]})

                # retry task
                user_retry = build_retry_prompt(q, reflection=refl["text"], dataset=dataset)
                assistant_retry = retry["solution"]
                rows.append({"messages":[{"role":"user","content":user_retry},{"role":"assistant","content":assistant_retry}]})

    write_jsonl(Path(args.out_jsonl), rows)
    print(f"Wrote {len(rows)} conversations to {args.out_jsonl}")

if __name__ == "__main__":
    main()
