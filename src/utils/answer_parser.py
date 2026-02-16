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

from fractions import Fraction
from decimal import Decimal, InvalidOperation

_BOXED_RE = re.compile(r"\\boxed\s*\{(.+?)\}", re.DOTALL)
_TEX_DOLLAR_RE = re.compile(r"^\s*\$\s*(.*?)\s*\$\s*$", re.DOTALL)
_TEX_PAREN_RE = re.compile(r"^\s*\\\(\s*(.*?)\s*\\\)\s*$", re.DOTALL)
_TEX_BRACKET_RE = re.compile(r"^\s*\\\[\s*(.*?)\s*\\\]\s*$", re.DOTALL)
_LEFT_RIGHT_RE = re.compile(r"\\left|\\right")
_FRAC_RE = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")
_SPACES_RE = re.compile(r"\s+")
_UNICODE_MINUS = "\u2212"

# numeric forms after normalization
_NUMERIC_RE = re.compile(r"^[\+\-]?\d+(\.\d+)?$")
_FRACTION_RE = re.compile(r"^[\+\-]?\d+\s*/\s*[\+\-]?\d+$")


def normalize_math_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()

    # Prefer content inside \boxed{...} if present
    m = _BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()

    # Strip outer TeX wrappers
    for pat in (_TEX_DOLLAR_RE, _TEX_PAREN_RE, _TEX_BRACKET_RE):
        m2 = pat.match(s)
        if m2:
            s = m2.group(1).strip()
            break

    s = s.replace(_UNICODE_MINUS, "-")
    s = _LEFT_RIGHT_RE.sub("", s)

    # Convert \frac{a}{b} -> (a)/(b) (simple, non-nested)
    while True:
        m3 = _FRAC_RE.search(s)
        if not m3:
            break
        a, b = m3.group(1).strip(), m3.group(2).strip()
        s = s[:m3.start()] + f"({a})/({b})" + s[m3.end():]

    # Strip outer braces if whole thing is {...}
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1].strip()

    # Remove commas inside numbers: 1,000 -> 1000
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    # Normalize whitespace
    s = _SPACES_RE.sub(" ", s).strip()
    return s


def try_parse_number(s: str):
    """Return Fraction for integers/decimals/fractions, else None."""
    if s is None:
        return None
    s = str(s).strip()

    # peel simple outer parentheses
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        if inner.count("(") == inner.count(")"):
            s = inner
        else:
            break

    # Convert (a)/(b) -> a/b if exactly that pattern
    s = re.sub(r"^\(\s*([^\(\)]+?)\s*\)\s*/\s*\(\s*([^\(\)]+?)\s*\)$", r"\1/\2", s)

    if _NUMERIC_RE.match(s):
        try:
            d = Decimal(s)
            return Fraction(d)
        except (InvalidOperation, ValueError):
            return None

    if _FRACTION_RE.match(s):
        try:
            num, den = s.split("/")
            return Fraction(int(num.strip()), int(den.strip()))
        except Exception:
            return None

    return None


def math_strict_equal(pred_raw: str, gold_raw: str) -> bool:
    pred = normalize_math_answer(pred_raw)
    gold = normalize_math_answer(gold_raw)

    pnum = try_parse_number(pred)
    gnum = try_parse_number(gold)
    if pnum is not None and gnum is not None:
        return pnum == gnum

    return pred == gold

def extract_math_final(text: str):
    """
    Extract final answer for MATH-style outputs.
    Priority:
      1) \boxed{...} (keep inside, not just a number)
      2) last line that looks like an answer-ish (fallback)
    """
    if not text:
        return None

    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    # fallback: try last "reasonable" token chunk
    # (keep it conservative)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


