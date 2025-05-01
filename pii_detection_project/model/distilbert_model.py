from transformers import AutoConfig, AutoModelForTokenClassification

def get_model(num_labels, id2label, label2id, model_name='distilbert-base-uncased'):
    """
    Instantiate a DistilBERT TokenClassification model.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config
    )
    return model
