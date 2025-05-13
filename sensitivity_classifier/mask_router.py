import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sensitivity_classifier import mask_text  # Import the masking function

# Load model and tokenizer only once at module level for efficiency
MODEL_PATH = "sensitivity_classifier/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def detect_pii(
    text: str,
    return_masked: bool = False,
    sensitive_labels: list = None
) -> dict:
    """
    Detects PII in the input text using the loaded model.
    - Only considers tags in sensitive_labels as PII. If sensitive_labels is None, all non-"O" tags are considered sensitive.
    - Returns a dict with detection status, original text, and optionally masked text.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    tags = [model.config.id2label[p] for p in preds]

    # For BERT-like models, the first and last tokens are usually special ([CLS], [SEP])
    real_token_tags = tags[1:-1] if len(tags) > 2 else tags

    # If no sensitive_labels provided, consider all non-"O" tags as sensitive
    if sensitive_labels is None:
        pii_detected = any(tag != "O" for tag in real_token_tags)
    else:
        pii_detected = any(tag in sensitive_labels for tag in real_token_tags)

    result = {
        "pii_detected": pii_detected,
        "original": text,
    }
    if return_masked and pii_detected:
        # Mask only the selected labels if provided
        result["masked"] = mask_text(text, sensitive_labels=sensitive_labels)
    return result

def handle_user_query(
    text: str,
    auto_choice: str = None,
    sensitive_labels: list = None
) -> dict:
    """
    Main entry point for routing a user query.
    - Detects PII and, if found, prompts the user (or uses auto_choice) to decide:
        [1] Route unmasked to LOCAL LLM
        [2] Mask and route to CLOUD LLM
    - sensitive_labels: list of tag strings to consider as sensitive (e.g., ["B-FIRSTNAME", "B-PHONENUMBER"]).
      For UI: present available labels (model.config.id2label.values()) as a checklist.
    - Returns a dict with routing info and the final text (masked or unmasked).
    """
    result = detect_pii(text, return_masked=True, sensitive_labels=sensitive_labels)

    if not result["pii_detected"]:
        # No PII detected, route to cloud by default
        return {
            "pii_detected": False,
            "routed_to": "cloud",
            "final_text": text
        }

    print("⚠️ Sensitive content detected.")
    if auto_choice not in ["local", "cloud"]:
        # Prompt user for routing choice in CLI
        print("\nChoose an option:")
        print("[1] Route unmasked to LOCAL LLM")
        print("[2] Mask it and route to CLOUD LLM")
        choice = input("Enter 1 or 2: ").strip()
    else:
        # Use provided auto_choice (for UI or automation)
        choice = "1" if auto_choice == "local" else "2"

    if choice == "1":
        # Route unmasked text to local LLM
        return {
            "pii_detected": True,
            "routed_to": "local",
            "final_text": text
        }
    else:
        # Mask PII and route to cloud LLM
        return {
            "pii_detected": True,
            "routed_to": "cloud",
            "final_text": result["masked"]
        }
