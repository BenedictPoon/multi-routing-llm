import argparse
from transformers import Trainer, TrainingArguments
from data.load_data import load_data
from data.preprocess import preprocess_dataset
from model.distilbert_model import get_model
from train.train import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    _, _, test_ds = load_data()
    unique = {lab for labels in test_ds['mbert_bio_labels'] for lab in labels}
    label_list = sorted(unique)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    tokenized_test = preprocess_dataset(
        test_ds,
        collapse_labels=False,
        label2id=label2id
    )

    model = get_model(num_labels=len(label2id),
                      id2label=id2label,
                      label2id=label2id,
                      model_name=args.model_dir)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == '__main__':
    main()