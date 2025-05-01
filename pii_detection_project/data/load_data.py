from datasets import load_dataset

def load_data():
    """
    Load the ai4privacy/pii-masking-200k dataset from Hugging Face.
    Returns train, validation, and test splits. If the dataset lacks a validation split,
    10% of the train split is held out as validation.
    """
    dataset = load_dataset("ai4privacy/pii-masking-200k")
    train_data = dataset["train"]
    test_data = dataset.get("test")
    if "validation" in dataset:
        val_data = dataset["validation"]
    else:
        # create a 90/10 train/validation split from train_data
        splits = train_data.train_test_split(test_size=0.1)
        train_data = splits["train"]
        val_data = splits["test"]
    return train_data, val_data, test_data
