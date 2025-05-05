from transformers import pipeline

def tag_text(text, model_dir='distilbert-base-uncased', aggregation_strategy=None):
    """
    Return token-level BIO tags for input text.
    """
    nlp = pipeline(
        'token-classification',
        model=model_dir,
        tokenizer=model_dir,
        aggregation_strategy=aggregation_strategy
    )
    output = nlp(text)
    return [(item['word'], item['entity']) for item in output]
