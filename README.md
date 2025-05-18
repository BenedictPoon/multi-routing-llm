# MultiLLM Query Router

A smart routing system that automatically directs queries to either local (Ollama) or cloud (OpenAI) models based on content sensitivity and user preferences.

## Features

- Automatic routing based on query sensitivity
- Support for both local (Ollama) and cloud (OpenAI) models
- Technical and general inquiry classification
- Beautiful Streamlit UI
- Comprehensive logging system

## Prerequisites

- Python 3.10 or higher
- Conda (for environment management)
- Ollama installed locally
- OpenAI API key

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-routing-llm.git
cd multi-routing-llm
```

2. Create and activate the conda environment:
```bash
conda create -n multillm python=3.10
conda activate multillm
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the sensitivity classifier model:
```bash
# Create the model directory
mkdir -p sensitivity_classifier/model

# Download the pre-trained model
# You can use a model from Hugging Face that's trained for NER (Named Entity Recognition)
# For example, using a model trained on sensitive information detection:
python -c "
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_name = 'your-preferred-model'  # e.g., 'dslim/bert-base-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer.save_pretrained('sensitivity_classifier/model')
model.save_pretrained('sensitivity_classifier/model')
"

# Create labels.json in the sensitivity_classifier directory
echo '{
    "0": "O",
    "1": "B-EMAIL",
    "2": "I-EMAIL",
    "3": "B-PHONE",
    "4": "I-PHONE",
    "5": "B-ACCOUNT",
    "6": "I-ACCOUNT",
    "7": "B-VRM",
    "8": "I-VRM"
}' > sensitivity_classifier/labels.json
```

5. Set up Ollama:
```bash
# Install Ollama if you haven't already
# Visit https://ollama.ai for installation instructions

# Pull the required model
ollama pull llama3
```

6. Set up your OpenAI API key:
```bash
# Either set it as an environment variable
export OPENAI_API_KEY='your-api-key'

# Or create a config.txt file in the project root
echo 'your-api-key' > config.txt
```

## Running the Application

1. Start the application:
```bash
python main.py
```

2. The Streamlit UI will open in your default web browser.

## Usage

1. Choose your preferred model:
   - **Auto**: Automatically selects between local and cloud models based on query sensitivity
   - **Cloud**: Always uses OpenAI's GPT model
   - **Local**: Always uses the local Ollama model

2. Enter your query in the text area and click "Submit"

3. The system will:
   - Classify your query as technical or general
   - Route it to the appropriate handler
   - Process it using the selected model
   - Display the response

## Model Training (Optional)

If you want to train your own sensitivity classifier:

1. Prepare your training data:
   - Collect examples of text with and without sensitive information
   - Label the sensitive entities (emails, phone numbers, etc.)
   - Format the data according to the Hugging Face dataset format

2. Fine-tune the model:
```bash
python train_classifier.py \
    --model_name "bert-base-uncased" \
    --train_file "path/to/train.json" \
    --validation_file "path/to/val.json" \
    --output_dir "sensitivity_classifier/model"
```

3. The trained model will be saved in the `sensitivity_classifier/model` directory.

## Logging

The application includes comprehensive logging that shows:
- Query processing steps
- Model selection decisions
- Classification results
- API calls and responses

Logs are displayed in the terminal where you run the application.

## Troubleshooting

### Common Issues

**Issue**: "OpenAI API key not set" error
**Solution**: Make sure you've set your API key as described in the Configuration section.

**Issue**: "Not in the correct conda environment" warning
**Solution**: Activate the multillm environment with `conda activate multillm`.

**Issue**: Streamlit not found
**Solution**: Make sure you've installed all requirements with `pip install -r requirements.txt`.

### Getting Help

If you continue to experience issues, check the following resources:
- Streamlit documentation: https://docs.streamlit.io/
- LangChain documentation: https://python.langchain.com/docs/
- OpenAI API documentation: https://platform.openai.com/docs/


# Masking Pipeline for PII Detection and Routing

This module provides a pipeline for fine-tuning a DistilBERT model to detect and mask PII (Personally Identifiable Information) in text, and for routing queries based on detected sensitivity.

---

## Project Structure

```
multi-routing-llm/
│
├── train.py
├── test_masking.py
└── sensitivity_classifier/
    ├── masking.py
    ├── mask_router.py
    ├── model/           # Saved model and tokenizer after training
    └── labels.json      # Label mapping used for masking
```

---

## Training

To train the model on the PII dataset:

```bash
python train.py
```

- Loads and preprocesses the dataset.
- Fine-tunes DistilBERT for token classification.
- Saves the trained model and label mappings to `sensitivity_classifier/model/` and `sensitivity_classifier/labels.json`.
- Make sure to adjust the hyperparameters for your machine, its currently set to CUDA (Nvidia), will not work on mac 

---

## Masking and Routing Usage

### Masking PII in Text

```python
from sensitivity_classifier import mask_text

text = "My name is John and my phone number is 0412 345 678."
masked = mask_text(text)
print(masked)
# Output: "My name is [FIRSTNAME] and my phone number is [PHONENUMBER]."
```

### Routing Based on PII Detection

```python
from sensitivity_classifier.mask_router import handle_user_query

result = handle_user_query("My name is John.", auto_choice="cloud")
print(result["routed_to"])    # "cloud"
print(result["final_text"])   # Masked or unmasked text
```

---

## Customizing Sensitive Labels

You can specify which entity types are considered sensitive and should be masked:

```python
labels = ["B-PHONENUMBER", "I-PHONENUMBER"]
result = handle_user_query(
    "My name is John and my phone number is 0412 345 678.",
    auto_choice="cloud",
    sensitive_labels=labels
)
print(result["final_text"])
# Output: "My name is John and my phone number is [PHONENUMBER]."
```

TRY ADD THIS: For UI integration, present `list(model.config.id2label.values())` as a checklist for user selection.

---

## Testing

A simple test is provided in `test_masking.py`:

```python
from sensitivity_classifier import mask_text

def test_basic():
    text = "My name is John and I live in New South Wales."
    out = mask_text(text)
    assert "[FIRSTNAME]" in out
```

---

## Requirements

- Python 3.8+
- transformers
- datasets
- seqeval
- torch

---