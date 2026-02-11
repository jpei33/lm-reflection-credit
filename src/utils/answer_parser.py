import re

# Matches integers, decimals, and simple fractions like 5/6
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def extract_final_answer_strict(text: str):
    """
    STRICT:
    Only accept answers that appear as:
        #### <number>
    (allows optional $ before number)
    """
    if not text:
        return None

    matches = re.findall(
        r"####\s*\$?\s*([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
        text,
    )
    return matches[-1].strip() if matches else None


def extract_final_answer_loose(text: str):
    """
    LOOSE:
    - #### <anything with a number>
    - \\boxed{<number>}
    - fallback: last number anywhere
    """
    if not text:
        return None

    # 1) #### line (keep last valid)
    last = None
    for m in re.finditer(r"####\s*([^\n\r]+)", text):
        s = m.group(1)
        m2 = _NUM_RE.search(s)
        if m2:
            last = m2.group(0).strip()
    if last is not None:
        return last

    # 2) boxed
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        m2 = _NUM_RE.search(m.group(1))
        if m2:
            return m2.group(0).strip()

    # 3) fallback: last number anywhere
    nums = _NUM_RE.findall(text)
    return nums[-1].strip() if nums else None
