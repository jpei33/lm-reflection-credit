import time
from typing import Dict, Any, Tuple
from src.utils.generator import GenConfig

class HFGenerator:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

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
        self.model.eval()

    def generate(self, prompt: str, cfg: GenConfig):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        dt = time.time() - t0

        # ✅ decode ONLY new tokens
        gen_ids = out[0, input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        meta = {"latency_s": dt, "model_name": self.model_name, "backend": "hf"}
        return text, meta

