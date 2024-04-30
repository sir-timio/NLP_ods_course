import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    set_seed,
)

import wandb

set_seed(42)


id2label = {
    0: "B-EMAIL",
    1: "B-ID_NUM",
    2: "B-NAME_STUDENT",
    3: "B-PHONE_NUM",
    4: "B-STREET_ADDRESS",
    5: "B-URL_PERSONAL",
    6: "B-USERNAME",
    7: "I-ID_NUM",
    8: "I-NAME_STUDENT",
    9: "I-PHONE_NUM",
    10: "I-STREET_ADDRESS",
    11: "I-URL_PERSONAL",
    12: "O",
}
label2id = {v: k for k, v in id2label.items()}
O_label_id = label2id["O"]


def load_json_datasets(paths: str):
    """load json dataframes"""
    dfs = []
    allowed_cols = ["tokens", "trailing_whitespace", "labels"]
    for path in paths:
        df = pd.read_json(path)[allowed_cols]
        df["ner_tags"] = df["labels"].apply(
            lambda labels_list: [label2id[x] for x in labels_list]
        )
        df["has_ents"] = df["labels"].apply(lambda labels: len(set(labels)) > 1)
        dfs.append(df)
    return pd.concat(dfs)


# Expand word labels to tokens labels
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples, max_length: int, tokenizer: AutoTokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    subtoken2word = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
        subtoken2word.append(word_ids)

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = subtoken2word
    return tokenized_inputs
