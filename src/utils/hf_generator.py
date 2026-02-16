import time
from src.utils.generator import GenConfig
import torch
import random

class HFGenerator:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Optional determinism knobs (helps on GPU)
        self.torch.backends.cudnn.deterministic = True
        self.torch.backends.cudnn.benchmark = False

        self.model.eval()

    def generate(self, prompt: str, cfg: GenConfig, seed: int = 0):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Deterministic sampling across runs (works even if transformers doesn't support generator=)
        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=(cfg.temperature > 0),
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        dt = time.time() - t0

        # decode ONLY new tokens
        gen_ids = out[0, input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        prompt_tokens = int(input_len)
        generated_tokens = int(gen_ids.shape[0])
        total_tokens = prompt_tokens + generated_tokens

        meta = {
            "latency_s": dt,
            "model_name": self.model_name,
            "backend": "hf",
            "seed": seed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
        }
        return text, meta
