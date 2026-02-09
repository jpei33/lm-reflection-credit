import re

FINAL_ANS_RE = re.compile(r"####\s*([-+]?\d*\.?\d+)")

def extract_final_answer(text: str):
    m = FINAL_ANS_RE.search(text)
    return m.group(1) if m else None
