import os
import time
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from src.utils.generator import GenConfig

class TinkerGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TINKER_API_KEY in environment (.env).")

        # Tinker SDK imports
        import tinker

        # NOTE: API shape can vary by SDK version; we’ll adjust if needed
        self.tinker = tinker
        self.client = tinker.Client(api_key=api_key)

    def generate(self, prompt: str, cfg: GenConfig) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()

        # Typical chat/instruct style payload:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
        )

        dt = time.time() - t0

        text = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
        meta = {"backend": "tinker", "model_name": self.model_name, "latency_s": dt}
        return text, meta
