import json, pathlib
from typing import Optional, List
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          pipeline)

HERE   = pathlib.Path(__file__).parent
MODEL  = HERE / "model"
LABELS = HERE / "labels.json"

with open(LABELS) as f:
    id2label = json.load(f)

PLACEHOLDERS = {lab.split("-")[-1]: f"[{lab.split('-')[-1]}]"
                for lab in id2label.values() if lab != "O"}

_masker = None

def load_masker(device:Optional[str]=None):
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
    out, shift = text, 0
    for ent in spans:
        s, e    = ent["start"], ent["end"]
        label   = ent["entity_group"].split("_")[-1]
        ph      = PLACEHOLDERS.get(label, "[MASK]")
        out     = out[:s+shift] + ph + out[e+shift:]
        shift  += len(ph) - (e-s)
    return out

def mask_text(text:str, device:Optional[str]=None) -> str:
    nlp   = load_masker(device)
    spans = nlp(text)
    return _replace(text, spans)
