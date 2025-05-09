#!/usr/bin/env python

"""
Fine-tune DistilBERT on ai4privacy/pii-masking-200k
and save model + tokenizer + labels.json into sensitivity_classifier/model/
"""
import json, itertools, os, numpy as np
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)
from sklearn.metrics import precision_score, recall_score, f1_score
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
OUT_DIR    = "sensitivity_classifier/model"

# 1 ─────────────────────────────────────────────────────── load + prune cols
raw = load_dataset("ai4privacy/pii-masking-200k", split="train")
raw = raw.remove_columns([
    "target_text", "privacy_mask", "mbert_text_tokens",
    "mbert_bio_labels", "set"
])
raw = raw.filter(lambda x: x["language"] == "en")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 2 ─────────────────────────────────────────────────────── label maps
all_spans = set()
for spans in raw["span_labels"]:
    for *_, lab in json.loads(spans):
        if lab != "O":
            all_spans.add(lab)

id2label = {0: "O"}
idx = 1
for lab in sorted(all_spans):        # all_spans already has no "O"
    id2label[idx] = f"B-{lab}"; idx += 1
    id2label[idx] = f"I-{lab}"; idx += 1

label2id = {v: k for k, v in id2label.items()}
id2tag = {i: t for i, t in id2label.items()}

missing = sorted(
    {f"{p}-{lab}" for lab in all_spans for p in ("B", "I")}
    - set(label2id)
)
assert not missing, f"Label mapping missing tags: {missing}"


with open("sensitivity_classifier/labels.json", "w") as f:
    json.dump(id2label, f)

# 3 ─────────────────────────────────────────────────────── encoder + align
def encode(example):
    text  = example["source_text"]

    spans_raw = [sp for sp in json.loads(example["span_labels"]) if sp[2] != "O"]
    spans = [(s, e, label2id[f"{pref}-{lab}"])
             for s, e, lab in spans_raw
             for pref in ("B", "I")]
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN,
                    return_offsets_mapping=True)
    # build default O labels
    tok_labels = np.zeros(len(enc["input_ids"]), dtype=int)
    # mark special tokens for loss masking later
    special = set([tokenizer.cls_token_id, tokenizer.sep_token_id])
    for i, (start, end) in enumerate(enc["offset_mapping"]):
        if enc["input_ids"][i] in special or start == end:
            tok_labels[i] = -100
            continue
        for s, e, lab in spans:
            if not (e <= start or s >= end):
                tok_labels[i] = lab
                break
    enc["labels"] = tok_labels.tolist()
    enc.pop("offset_mapping")
    return enc

proc = raw.map(encode, batched=False, remove_columns=raw.column_names,
               num_proc=os.cpu_count())
proc.set_format("torch")
split = proc.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

#4 ───────────────────────────────────────────────────────  compute metrics
def align_and_strip(pred, labels):
    """Remove the -100 padding tokens and convert IDs → tag strings."""
    true_tags, pred_tags = [], []
    for p_row, l_row in zip(pred, labels):
        true_seq, pred_seq = [], []
        for p_id, l_id in zip(p_row, l_row):
            if l_id == -100:                     # ignore special / sub-tokens
                continue
            true_seq.append(id2tag[l_id])
            pred_seq.append(id2tag[p_id])
        true_tags.append(true_seq)
        pred_tags.append(pred_seq)
    return true_tags, pred_tags


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    true_tags, pred_tags = align_and_strip(preds, labels)

    return {
        "precision": precision_score(true_tags, pred_tags),
        "recall"   : recall_score(true_tags, pred_tags),
        "f1"       : f1_score(true_tags, pred_tags)
    }


# 4 ─────────────────────────────────────────────────────── training
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id),
    id2label=id2label, label2id=label2id)
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,      # ↓ from 8
    gradient_accumulation_steps=4,      # keeps effective batch = 8
    num_train_epochs=3,
    bf16=True,                          # Apple-GPU
    use_mps_device=True,                # M-series macs
    eval_strategy="epoch",
    # fp16=True,                         # Cuda-GPU
    # eval_strategy="epoch",
    # save_strategy="epoch",
    # logging_steps=100
)
collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(model=model, args=args,
                  data_collator=collator,
                  train_dataset=train_ds,
                  eval_dataset=val_ds,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("✓ Model saved to", OUT_DIR)


