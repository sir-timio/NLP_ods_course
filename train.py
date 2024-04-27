# %%
import os

os.environ["WANDB_PROJECT"] = "PII Data Detection"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "all"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# %%
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from scipy.special import softmax
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import wandb

# torch.autograd.set_detect_anomaly(True)

random_seed = 42
set_seed(random_seed)

INFERENCE_MAX_LENGTH = 128
wandb_run_name = f"deberta-base-{INFERENCE_MAX_LENGTH}-focal"
# wandb_run_name = "deleteme"
model_save_path = f"/archive/savkin/models/ner/PII Data Detection/{wandb_run_name}"

# %% [markdown]
# # Load dataset

# %%
# Load dataset and convert ner_tags to labels
allowed_cols = [
    "document",
    "full_text",
    "tokens",
    "trailing_whitespace",
    "labels",
    "valid",
]

df = pd.read_json(
    "/archive/savkin/parsed_datasets/NER/PII_Data_Detection/orig_train_custom_split.json"
)[allowed_cols]

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


df["ner_tags"] = df["labels"].apply(
    lambda labels_list: [label2id[x] for x in labels_list]
)
df["has_ents"] = df["labels"].apply(lambda labels: len(set(labels)) > 1)

train_df = df[df["valid"] == False].reset_index()
valid_df = df[df["valid"] == True].reset_index()

dataset = DatasetDict(
    {"train": Dataset.from_pandas(train_df), "valid": Dataset.from_pandas(valid_df)}
)

# %% [markdown]
# # Tokenize data

# %%
# Load model and tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
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


# Tokenize dataset and align labels with tokens
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=INFERENCE_MAX_LENGTH,
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


tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
tokenized_dataset["train"].features

# %% [markdown]
# # Train model

# %%
seqeval_metrics = evaluate.load("seqeval")


def f5_score(precision, recall):
    return (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall + 1e-100)


def compute_metrics_from_labels(predictions, labels):
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metrics.compute(
        predictions=true_predictions, references=true_labels
    )
    for label, scores in results.items():
        if "overall" not in label:
            precision = scores["precision"]
            recall = scores["recall"]
            results[label]["f5_score"] = f5_score(precision, recall)
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    results["overall_f5_score"] = f5_score(precision, recall)

    return results


def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)

    return compute_metrics_from_labels(predictions, labels)


def compute_metrics_crf(eval_preds):
    tags, labels = eval_preds
    return compute_metrics_from_labels(tags, labels)


# %%
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, pad_to_multiple_of=16
)

# %%
import sys

sys.path.append("..")
from src.deberta_crf import DebertaV2WithCRF
from src.deberta_focal import DebertaV2FocalLoss
from src.deberta_lstm import DebertaV2WithLSTM
from src.deberta_lstm_crf import DebertaV2WithLSTMCRF

# %%
# model = DebertaV2WithCRF.from_pretrained(
# model = AutoModelForTokenClassification.from_pretrained(
model = DebertaV2WithCRF.from_pretrained(
    # model = DebertaV2WithLSTMCRF.from_pretrained(
    # model = DebertaV2WithLSTM.from_pretrained(
    # model = DebertaV2FocalLoss.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
)
model.to("cuda")
# model.crf.to('cuda')

BATCH_SIZE = 2

# %%
training_args = TrainingArguments(
    output_dir="training_logs",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=5,
    # num_train_epochs=1,
    # max_steps=400,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    warmup_steps=600,
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    save_total_limit=1,
    metric_for_best_model="overall_f5_score",
    greater_is_better=True,
    load_best_model_at_end=True,
    # report_to="wandb",
    report_to="none",
    run_name=wandb_run_name,
    # max_grad_norm=1,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_crf,
    # compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(4)]
)

# %%
train_loader = trainer.get_train_dataloader()

# %%
trainer.train()
wandb.finish()

trainer.save_model(model_save_path)

# # %% [markdown]
# # ## Post-Evaluation

# # %%
# # Load model from saved if needed

# model_checkpoint = "/archive/savkin/models/ner/PII Data Detection/deberta-base-4000-lstm"
model_checkpoint = model_save_path
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
trainer = Trainer(
    args=TrainingArguments(output_dir="tmp_trainer", report_to="none"),
    model=model,
    data_collator=data_collator,
)

# # %%
run = wandb.init(name=f"{wandb_run_name}-post-evaluation", job_type="post-evaluation")

# # %%
predictions = trainer.predict(tokenized_dataset["valid"])

preds = predictions.predictions
true_labels = predictions.label_ids
metrics = compute_metrics_from_labels(preds, true_labels)
for metric_name, metric in metrics.items():
    run.log({metric_name: metric})

# # %% [markdown]
# # ## Log metrics depending on the threshold

# # %%
# thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
# # thresholds = [0.5, 0.6, 0.7]

# thresholded_metrics = {}
# best_threshold = 0
# for i, threshold in enumerate(thresholds):
# preds = predictions.predictions
# true_labels = predictions.label_ids
#     thresholed_pred_labels = apply_threshold(preds, threshold, O_label_id)

#     metrics = compute_metrics_from_labels(thresholed_pred_labels, true_labels)
#     thresholded_metrics[threshold] = metrics
#     f5 = metrics["overall_f5_score"]
#     print(f"Threshold {threshold}, overall_f5_score = {f5}")

# # %%
# # Log metrics based on threshold
# run.define_metric("threshold")
# run.define_metric(f"thresholded_*", step_metric="threshold", summary="max")

# for threshold, metrics in thresholded_metrics.items():
#     for metric_name, metric in metrics.items():
#         new_metric_name = f"thresholded_{metric_name}"
#         run.log({new_metric_name: metric, "threshold": threshold})
#         # print({new_metric_name: metric, "threshold": threshold})

# thresholed_f5_scores = [metric["overall_f5_score"] for _, metric in thresholded_metrics.items()]
# best_threshold_id = np.argmax(thresholed_f5_scores)
# best_threshold = thresholds[best_threshold_id]
# run.summary["best_overall_threshold"] = best_threshold
# run.summary["best_overall_f5_score"] = thresholed_f5_scores[best_threshold_id]

# # %% [markdown]
# # ## Aggregate subtoken-level predictions into word-level predictions

# # %%
# pred_probas = softmax(predictions.predictions, axis=-1).max(-1)
# pred_labels = apply_threshold(predictions.predictions, best_threshold, O_label_id)
# true_labels = predictions.label_ids

# # %%
# submission = {
#     "row_id": [],
#     "document": [],
#     "token": [],
#     "label": [],
#     "subtoken_str": [],
#     "word_str": [],
#     "proba": []
# }

# for input_ids, word_ids, row_id, document, words, p_labels, p_probas in zip(tokenized_dataset["valid"]["input_ids"],
#                                                                             tokenized_dataset["valid"]["word_ids"],
#                                                                             valid_df.index,
#                                                                             valid_df["document"],
#                                                                             valid_df["tokens"],
#                                                                             pred_labels,
#                                                                             pred_probas):
#     subtokens = tokenizer.convert_ids_to_tokens(input_ids)
#     for subtoken_id, (subtoken, label_id, proba) in enumerate(zip(subtokens, p_labels, p_probas)):
#         word_id = word_ids[subtoken_id]
#         if label_id != -100 and label_id != O_label_id and word_id is not None: # ignore O-labels
#             submission["row_id"].append(row_id)
#             submission["document"].append(document)
#             submission["token"].append(word_id)
#             submission["label"].append(id2label[label_id])
#             submission["subtoken_str"].append(subtoken)
#             submission["word_str"].append(words[word_id])
#             submission["proba"].append(proba)


# df = pd.DataFrame().from_dict(submission).drop_duplicates().sort_values(by=["document", "token"])
# # submission_df_subtoken_level = df[df["label"] != "O"].copy(deep=True)[["row_id", "document", "token", "label", "proba"]]

# subtoken_df =  df[df["label"] != "O"].copy(deep=True)
# subtoken_df.head()

# # %%
# def aggregate_subtokens(df, label_agg_type = "most_frequent", add_subtoken_info = False):
#     df = df.reset_index()
#     row = df.iloc[0]

#     if add_subtoken_info:
#         row["subtokens"] = df["subtoken_str"].agg(lambda x: x.tolist())
#         row["probas"] = df["proba"].agg(lambda x: x.tolist())

#     if label_agg_type == "most_frequent":
#         row["label"] = df.groupby(["label"])["row_id"].count().sort_values().index[-1]
#         row["agg_proba"] = df[df["label"] == row["label"]]["proba"].agg("mean")
#     elif label_agg_type == "first":
#         row["label"] = df["label"].agg(lambda x: x[0])
#         row["agg_proba"] = df["proba"].agg(lambda x: x[0])
#     elif label_agg_type == "max_proba":
#         row["label"] = df.iloc[df["proba"].idxmax()]["label"]
#         row["agg_proba"] = df["proba"].agg("max")

#     return row

# # submission_df = submission_df_subtoken_level.groupby(["document", "token"]) \
# #                                             .apply(aggregate_subtokens, label_agg_type="most_frequent") \
# #                                             .reset_index(drop=True) \
# #                                             .drop(columns=["index", "proba", "agg_proba"])

# word_df = subtoken_df.groupby(["document", "token"]) \
#                      .apply(aggregate_subtokens, add_subtoken_info=True) \
#                      .reset_index(drop=True) \
#                      .drop(columns=["index", "subtoken_str", "proba"])
# word_df.head()

# # %% [markdown]
# # ## Logging word-level predictions

# # %% [markdown]
# # ## Log word-level metrics

# # %%
# true_word_labels = valid_df["labels"].apply(lambda labels: [label2id[l] for l in labels]).tolist()

# # Create a template filled with "O" label
# pred_word_labels = valid_df["labels"].agg(lambda x: [O_label_id for _ in x]).tolist()

# # Group words into documents and reorder documents according to validation dataset
# original_document_order = valid_df["document"].tolist()
# document_df = word_df[["document", "token", "label"]].groupby("document").agg(list)
# reordered_document_df = document_df.reindex(original_document_order, fill_value=[])

# # Add predictions to the template
# for i, (_, row) in enumerate(reordered_document_df.iterrows()):
#     if len(row["token"]) > 0:
#         for token_id, l in zip(row["token"], row["label"]):
#             pred_word_labels[i][token_id] = label2id[l]

# word_level_metrics = compute_metrics_from_labels(pred_word_labels, true_word_labels)

# # %%
# run.define_metric(f"word_level*")
# for metric_name, metric in word_level_metrics.items():
#     new_metric_name = f"word_level_{metric_name}"
#     run.summary[new_metric_name] = metric

# # %% [markdown]
# # ## Log model mistakes

# # %%
# error_rows = []
# for (_, valid_row), pred_doc_labels, true_doc_labels in zip(valid_df.iterrows(), pred_word_labels, true_word_labels):

#     pred_doc_labels = np.array(pred_doc_labels)
#     true_doc_labels = np.array(true_doc_labels)
#     errors_mask = pred_doc_labels != true_doc_labels

#     if sum(errors_mask) == 0:
#         continue

#     words = (valid_row["tokens"])
#     trailing_whitespaces = valid_row["trailing_whitespace"]
#     doc_id = valid_row["document"]


#     error_pred_labels = pred_doc_labels[errors_mask]
#     error_true_labels = true_doc_labels[errors_mask]
#     error_words = np.array(words)[errors_mask]
#     error_word_ids = np.argwhere(errors_mask)


#     target_vizualization = wandb.Html(visualize_ents(words, trailing_whitespaces, [id2label[l] for l in true_doc_labels]))
#     pred_vizualization = wandb.Html(visualize_ents(words, trailing_whitespaces, [id2label[l] for l in pred_doc_labels]))

#     row = {}
#     for w, w_id, p_l, t_l in zip(error_words, error_word_ids, error_pred_labels, error_true_labels):
#         w_id = w_id[0]
#         pred_row = word_df[word_df["document"] == doc_id]
#         pred_row = pred_row[pred_row["token"] == w_id]

#         row["document"] = doc_id
#         row["word"] = w
#         row["word_id"] = w_id
#         row["pred_label"] = id2label[p_l]
#         row["true_label"] = id2label[t_l]
#         row["target_viz"] = target_vizualization
#         row["pred_viz"] = pred_vizualization

#         assert len(pred_row) <= 1

#         if len(pred_row) == 1:
#             row["subtokens"] = pred_row["subtokens"].to_numpy().squeeze()
#             row["probas"] = pred_row["probas"].tolist()[0]
#             row["agg_proba"] = pred_row["agg_proba"].tolist()[0]
#         elif len(pred_row) == 0:
#             row["subtokens"] = None
#             row["probas"] = None
#             row["agg_proba"] = None
#     error_rows.append(row)


# error_df = pd.DataFrame().from_records(error_rows).sort_values(by=["document", "word_id"])
# error_df.head()

# # %%
# error_table = wandb.Table(dataframe=error_df)
# run.summary["error_table"] = error_table

# # %%
# wandb.finish()
