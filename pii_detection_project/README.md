"""
# PII Detector

This project fine-tunes DistilBERT on the `ai4privacy/pii-masking-200k` dataset
for token-level PII detection using a BIO tagging scheme.

## Setup

```bash
# Create environment (conda or venv)
pip install -r requirements.txt
```

## Directory Structure

- `data/`: loading and preprocessing dataset
- `models/`: model instantiation
- `train/`: training script with TensorBoard logging
- `eval/`: evaluation script
- `inference/`: simple wrapper for inference
- `utils/`: helper functions

## Usage

1. Train:
   ```bash
   python train/train.py \
     --output_dir ./outputs \
     --epochs 3 \
     --batch_size 16 \
     --learning_rate 5e-5
   ```
2. Evaluate:
   ```bash
   python eval/evaluate.py \
     --model_dir ./outputs \
     --batch_size 16
   ```
3. Inference (in Python):
   ```python
   from inference.inference import tag_text
   tags = tag_text("Alice’s phone number is 123-456-7890.")
   print(tags)
   ```
"""