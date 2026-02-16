from src.utils.answer_parser import math_strict_equal, extract_math_final

def test_math_equiv():
    assert math_strict_equal(r"\boxed{\frac{1}{2}}", "0.5")
    assert math_strict_equal(r"$\frac{2}{4}$", r"\boxed{1/2}")
    assert math_strict_equal("1,000", "1000")

def test_extract_math_boxed():
    out = "some work...\n\\boxed{\\frac{1}{2}}\n"
    assert extract_math_final(out) == r"\frac{1}{2}"
