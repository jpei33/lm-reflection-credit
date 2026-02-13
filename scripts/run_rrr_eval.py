import argparse
from dotenv import load_dotenv

from src.rrr.rrr_infer import run_rrr_eval
from src.utils.generator import GenConfig

# Load environment variables from .env file
load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "tinker"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--input", dest="input_jsonl", default="data/processed/gsm8k_train.jsonl")
    ap.add_argument("--output", dest="output_jsonl", default="results/runs/rrr_eval.jsonl")

    # Optional decoding knobs
    ap.add_argument("--solve_max_new_tokens", type=int, default=256)
    ap.add_argument("--retry_max_new_tokens", type=int, default=256)
    ap.add_argument("--reflect_max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--reflect_temperature", type=float, default=0.3)
    ap.add_argument("--reflect_top_p", type=float, default=0.9)

    args = ap.parse_args()

    if args.backend == "hf":
        from src.utils.hf_generator import HFGenerator
        gen = HFGenerator(args.model)
    else:
        from src.utils.tinker_generator import TinkerGenerator
        gen = TinkerGenerator(args.model)

    run_rrr_eval(
        gen=gen,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        limit=args.limit,
        solve_cfg=GenConfig(
            max_new_tokens=args.solve_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        reflect_cfg=GenConfig(
            max_new_tokens=args.reflect_max_new_tokens,
            temperature=args.reflect_temperature,
            top_p=args.reflect_top_p,
        ),
        retry_cfg=GenConfig(
            max_new_tokens=args.retry_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )


if __name__ == "__main__":
    main()
