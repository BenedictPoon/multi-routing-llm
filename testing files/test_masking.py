from sensitivity_classifier import mask_text

def test_basic():
    out = mask_text("My name is John   Smith.")
    assert "[FIRSTNAME]" in out or "[SURNAME]" in out