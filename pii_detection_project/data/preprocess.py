from transformers import AutoTokenizer

def preprocess_dataset(dataset, tokenizer_name='distilbert-base-uncased',
                       max_length=128, collapse_labels=False, label2id=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Build collapse_map if needed (maps old_id→new_id ints)
    if collapse_labels and label2id:
        collapse_map = {}
        for lbl, idx in label2id.items():
            if lbl == 'O': collapse_map[idx] = 0
            else:          collapse_map[idx] = 1 if lbl.startswith('B-') else 2
    else:
        collapse_map = None

    def tokenize_and_align(examples):
        # 1) map strings→IDs for every label sequence
        raw_label_seqs = examples['mbert_bio_labels']
        seqs_as_ids = [
            [ label2id[tag] for tag in tags ]
            for tags in raw_label_seqs
        ]
        # apply collapse if requested
        if collapse_map:
            seqs_as_ids = [
                [ collapse_map[id_] for id_ in seq ]
                for seq in seqs_as_ids
            ]

        # 2) tokenize
        tokenized = tokenizer(
            examples['mbert_text_tokens'],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        # 3) align those integer labels to sub-tokens
        aligned_labels = []
        for i, label_seq in enumerate(seqs_as_ids):
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            ids = []
            for wid in word_ids:
                if wid is None:
                    ids.append(-100)
                elif wid != prev:
                    ids.append(label_seq[wid])
                    prev = wid
                else:
                    ids.append(-100)
            aligned_labels.append(ids)

        tokenized['labels'] = aligned_labels
        return tokenized

    return dataset.map(
        tokenize_and_align,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
