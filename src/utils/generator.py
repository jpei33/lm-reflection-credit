from dataclasses import dataclass
from typing import Dict, Any, Tuple, Protocol

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

class Generator(Protocol):
    def generate(self, prompt: str, cfg: GenConfig) -> Tuple[str, Dict[str, Any]]:
        ...
