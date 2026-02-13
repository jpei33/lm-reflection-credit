import json
import time
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

from src.utils.answer_parser import extract_final_answer_strict


def build_prompt(question: str) -> str:
    # Force the "#### <answer>" format so our parser can extract it.
    return (
        "You are a helpful math tutor. Solve the problem step by step.\n"
        "At the end, output the final answer on a single line in the format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n\n"
        "Solution:\n"
    )


def make_hf_generator(model_name: str):
    """
    Returns a function generate(question)->(text, meta)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def generate(question: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95):
        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Keep only the part after "Solution:" (avoid repeating prompt in logs)
        if "Solution:\n" in text:
            text = text.split("Solution:\n", 1)[1].strip()

        meta = {"latency_s": dt, "model_name": model_name}
        return text, meta

    return generate


def run_rollouts(
    input_jsonl: str,
    output_jsonl: str,
    limit: int = 50,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
):
    """
    Loads GSM8K JSONL (question + answer), generates a solution, parses final answer,
    compares to ground truth, and writes rollouts JSONL.
    """
    gen = make_hf_generator(model_name)

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_correct = 0

    with open(input_jsonl, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= limit:
                break

            ex = json.loads(line)
            question = ex["question"]
            gt_solution = ex["answer"]
            gt_final = extract_final_answer_strict(gt_solution)

            model_solution, meta = gen(question)
            pred_final = extract_final_answer_strict(model_solution)

            correct = (pred_final == gt_final)
            n += 1
            n_correct += int(correct)

            record = {
                "question": question,
                "gt_final": gt_final,
                "pred_final": pred_final,
                "correct": correct,
                "model_solution": model_solution,
                "meta": meta,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {n} rollouts to {out_path}")
    print(f"Accuracy (parsed final answers): {n_correct}/{n} = {n_correct/max(n,1):.3f}")
