import time
from typing import Dict, Any, Tuple

from src.utils.generator import GenConfig


class TinkerGenerator:
    """
    Minimal generator wrapper around Tinker SamplingClient.

    Uses the Tinker Python SDK:
      - ServiceClient() -> create_lora_training_client(...)
      - training_client.save_weights_and_get_sampling_client(...)
      - sampling_client.sample(...)
    """
    def __init__(self, base_model: str, rank: int = 8, sampler_name: str = "sampler"):
        import os
        import tinker
        from tinker import types

        api_key = os.getenv("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TINKER_API_KEY in environment (.env or shell).")

        # ServiceClient reads TINKER_API_KEY from env.
        self.tinker = tinker
        self.types = types
        self.base_model = base_model

        self.service = tinker.ServiceClient()

        # Create a LoRA training client (even for eval-only).
        # This is the standard entrypoint shown in the docs. :contentReference[oaicite:2]{index=2}
        self.training_client = self.service.create_lora_training_client(
            base_model=base_model,
            rank=rank,
        )

        # Tokenizer for encode/decode
        self.tokenizer = self.training_client.get_tokenizer()

        # Create a sampler checkpoint and get a SamplingClient.
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=sampler_name
        )

    def generate(self, prompt: str, cfg: GenConfig) -> Tuple[str, Dict[str, Any]]:
        # Encode prompt into a ModelInput
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = self.types.ModelInput.from_ints(tokens=prompt_tokens)

        params = self.tinker.SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            # stop=["\n\n"]  # optional; you can add stops later
        )

        t0 = time.time()
        # One sample
        res = self.sampling_client.sample(
            prompt=model_input,
            sampling_params=params,
            num_samples=1,
        ).result()
        dt = time.time() - t0

        # Decode generated tokens
        out_tokens = res.sequences[0].tokens
        text = self.tokenizer.decode(out_tokens).strip()

        meta = {
            "backend": "tinker",
            "model_name": self.base_model,
            "latency_s": dt,
        }
        return text, meta
