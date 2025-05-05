import argparse
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification, AutoTokenizer, set_seed
from sklearn.metrics import precision_recall_fscore_support as prfs
from datasets import DatasetDict
from data.load_data import load_data
from data.preprocess import preprocess_dataset
from model.distilbert_model import get_model
from utils.logger import get_logger

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        for pred, lab in zip(pred_row, label_row):
            if lab != -100:
                true_labels.append(lab)
                true_preds.append(pred)
    precision, recall, f1, _ = prfs(true_labels, true_preds, average='micro')
    return {'precision': precision, 'recall': recall, 'f1': f1}

set_seed(42)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--collapse_labels', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit number of training samples for quick tests')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='Limit number of eval samples for quick tests')
    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info("Loading dataset...")
    train_ds, val_ds, _ = load_data()

    # Optionally limit raw dataset size
    if args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples:
        val_ds = val_ds.select(range(args.max_eval_samples))

    # Build label mappings
    logger.info("Building label list...")
    feature = train_ds.features["mbert_bio_labels"]
    if args.collapse_labels:
        label_list = ['O', 'B-PII', 'I-PII']
    else:
        # Gather unique labels from both train and validation datasets
        unique = {lab for ds in [train_ds, val_ds] for labels in ds['mbert_bio_labels'] for lab in labels}
        label_list = sorted(unique)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    logger.info("Tokenizing and aligning labels...")
    tokenized_train = preprocess_dataset(
        train_ds,
        collapse_labels=args.collapse_labels,
        label2id=label2id
    )
    tokenized_val = preprocess_dataset(
        val_ds,
        collapse_labels=args.collapse_labels,
        label2id=label2id
    )

    logger.info("Initializing model...")
    model = get_model(num_labels=len(label2id),
                      id2label=id2label,
                      label2id=label2id)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        logging_strategy='steps',
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=3,                # only keep your 3 best checkpoints
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to=['tensorboard']
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,              # add tokenizer
        data_collator=data_collator,      # add dynamic padding
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()