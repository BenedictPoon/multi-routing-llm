import json, pathlib
from typing import Optional, List, Tuple
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          pipeline)

# Set up paths for model and label mapping
HERE   = pathlib.Path(__file__).parent
MODEL  = HERE / "model"
LABELS = HERE / "labels.json"

# Load label mapping (id2label) from JSON
with open(LABELS) as f:
    id2label = json.load(f)

# Create a dictionary mapping label types to placeholder strings, e.g. "FIRSTNAME" -> "[FIRSTNAME]"
PLACEHOLDERS = {lab.split("-")[-1]: f"[{lab.split('-')[-1]}]"
                for lab in id2label.values() if lab != "O"}

_masker = None

def load_masker(device:Optional[str]=None):
    """
    Loads the Hugging Face pipeline for token classification (PII detection).
    Uses a singleton pattern to avoid reloading the model/tokenizer.
    """
    global _masker
    if _masker is None:
        tok   = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(MODEL)
        if device:
            model.to(device)
        _masker = pipeline("token-classification",
                           model=model, tokenizer=tok,
                           aggregation_strategy="simple")
    return _masker

def _replace(text:str, spans:List[dict]) -> str:
    """
    Replaces each detected sensitive span in the text with its corresponding placeholder.
    Handles shifting indices as replacements are made.
    """
    out, shift = text, 0
    for ent in spans:
        s, e    = ent["start"], ent["end"]
        label   = ent["entity_group"].split("_")[-1]
        ph      = PLACEHOLDERS.get(label, "[MASK]")
        out     = out[:s+shift] + ph + out[e+shift:]
        shift  += len(ph) - (e-s)
    return out

def _merge_spans(spans: List[dict]) -> List[dict]:
    """
    Merge overlapping or adjacent spans with the same label.
    This prevents repeated or fragmented masking of the same entity.
    The bug that we were trying to fix where the same entity was masked multiple times
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x["start"])
    merged = [spans[0]]
    for curr in spans[1:]:
        prev = merged[-1]
        if curr["start"] <= prev["end"] and curr["entity_group"] == prev["entity_group"]:
            prev["end"] = max(prev["end"], curr["end"])
        else:
            merged.append(curr)
    return merged

def mask_text(text: str, device: Optional[str] = None, sensitive_labels: list = None) -> str:
    """
    Mask only the spans whose entity_group is in sensitive_labels.
    If sensitive_labels is None, mask all non-'O' entities.
    """
    nlp = load_masker(device)
    spans = nlp(text)
    spans = _merge_spans(spans)
    if sensitive_labels is not None:
        # Only keep spans whose entity_group (label) is in sensitive_labels
        spans = [sp for sp in spans if sp["entity_group"] in sensitive_labels]
    return _replace(text, spans)

def check_sensitivity(text:str, device:Optional[str]=None) -> Tuple[bool, str]:
    """
    Check if text contains sensitive information and return masked version.
    
    Args:
        text (str): The text to check
        device (Optional[str]): Device to run the model on (e.g., 'cuda', 'cpu')
        
    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if sensitive information was found and masked
            - str: The masked text if sensitive info was found, original text otherwise
    """
    nlp = load_masker(device)
    spans = nlp(text)
    
    # If no spans were found, no sensitive information
    if not spans:
        return False, text
        
    # If spans were found, text contains sensitive information
    masked_text = _replace(text, spans)
    return True, masked_text
