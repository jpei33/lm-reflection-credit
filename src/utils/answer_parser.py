import re
from fractions import Fraction
from decimal import Decimal, InvalidOperation

# ----------------------------
# Shared numeric token patterns
# ----------------------------

# One "number token" that supports:
#  - integers: 12
#  - decimals: -3.5
#  - simple fractions: 5/6, -1/2, 3.0/4, 3/4.0
_NUM_TOKEN = r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"

# Find number tokens anywhere
_NUM_RE = re.compile(_NUM_TOKEN)

# STRICT marker line: #### <number> (optional $)
_HASH_STRICT_RE = re.compile(rf"####\s*\$?\s*({_NUM_TOKEN})")

# LOOSE: any #### line
_HASH_LINE_RE = re.compile(r"####\s*([^\n\r]+)")

# LOOSE: boxed
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


def extract_final_answer_strict(text: str):
    """
    STRICT:
    Only accept answers that appear as:
        #### <number>
    (allows optional $ before number)
    """
    if not text:
        return None

    matches = _HASH_STRICT_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_final_answer_loose(text: str):
    """
    LOOSE final-answer extractor.

    Priority:
      1) Last "#### <number>" line (GSM8K-style)
      2) Last "\boxed{...}" (supports multiple; grabs the last)
      3) Fallback: last number appearing near the end of the text (last 5 non-empty lines)
    """
    if not text:
        return None

    # 1) #### line (keep last match)
    m = None
    for mm in _HASH_LINE_RE.finditer(text):
        m = mm
    if m:
        line = m.group(0)
        nm = _NUM_RE.search(line)
        return nm.group(0) if nm else None

    # 2) boxed (keep last)
    m = None
    for mm in _BOXED_RE.finditer(text):
        m = mm
    if m:
        inside = m.group(1).strip()
        nm = _NUM_RE.search(inside)
        return nm.group(0) if nm else inside

    # 3) fallback: last number in the last few lines (avoid grabbing early intermediate values)
    tail_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][-5:]
    tail = "\n".join(tail_lines)
    nums = list(_NUM_RE.finditer(tail))
    return nums[-1].group(0) if nums else None


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
    s = re.sub(
        r"^\(\s*([^\(\)]+?)\s*\)\s*/\s*\(\s*([^\(\)]+?)\s*\)$",
        r"\1/\2",
        s,
    )

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
      2) last non-empty line (fallback)
    """
    if not text:
        return None

    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None
