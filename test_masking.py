from sensitivity_classifier import mask_text
from sensitivity_classifier.mask_router import handle_user_query

text = "What is Christian Bale know for?"
result = handle_user_query(text)

print(f"→ Routed to: {result['routed_to']}")
print(f"→ Final text: {result['final_text']}")

def test_basic():
    text  = "My name is John and I live in New South Wales."
    out   = mask_text(text)

    # show the transformation
    print(f"\nINPUT : {text}\nOUTPUT: {out}")

    # make sure we caught at least one name token
    assert "[FIRSTNAME]" in out