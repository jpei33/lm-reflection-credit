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
        print("MODEL DEVICE:", next(self.model.parameters()).device)    

        self.model.eval()

    def generate(self, prompt, cfg: GenConfig, seed: int = 0):
        """
        prompt: str OR List[str]
        returns:
        - if str: (text, meta)
        - if list: (texts, metas)
        """
        is_batched = isinstance(prompt, (list, tuple))
        prompts = list(prompt) if is_batched else [prompt]

        # Tokenize with padding so we can batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )

        if self.torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Deterministic sampling
        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

        # Per-example true prompt lengths (number of non-pad tokens)
        # attention_mask is 1 for real tokens, 0 for padding
        attn = inputs.get("attention_mask", None)
        if attn is None:
            # fallback (shouldn't happen if return_attention_mask=True)
            prompt_lens = [inputs["input_ids"].shape[1]] * len(prompts)
        else:
            prompt_lens = attn.sum(dim=1).tolist()  # list of ints

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

        # Decode per example: slice off its prompt tokens
        texts = []
        metas = []

        # out shape: (B, T_out)
        for i in range(out.shape[0]):
            input_len_i = int(prompt_lens[i])
            gen_ids = out[i, input_len_i:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            prompt_tokens = input_len_i
            generated_tokens = int(gen_ids.shape[0])
            total_tokens = prompt_tokens + generated_tokens

            meta = {
                "latency_s": dt / max(len(prompts), 1),  # per-example approx
                "batch_latency_s": dt,                   # full batch time
                "batch_size": len(prompts),
                "model_name": self.model_name,
                "backend": "hf",
                "seed": seed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": total_tokens,
            }
            texts.append(text)
            metas.append(meta)

        return (texts, metas) if is_batched else (texts[0], metas[0])
