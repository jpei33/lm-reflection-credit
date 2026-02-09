from src.rrr.rollout import run_rollouts

if __name__ == "__main__":
    run_rollouts(
        input_jsonl="data/processed/gsm8k_train.jsonl",
        output_jsonl="results/runs/rollout_hf_test.jsonl",
        limit=25,
    )
