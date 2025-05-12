from sensitivity_classifier import mask_text

def test_basic():
    text  = "My name is John and I live in New South Wales."
    out   = mask_text(text)

    # show the transformation
    print(f"\nINPUT : {text}\nOUTPUT: {out}")

    # make sure we caught at least one name token
    assert "[FIRSTNAME]" in out