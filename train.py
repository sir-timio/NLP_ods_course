import json
import logging
import os
import sys

os.environ["WANDB_PROJECT"] = "PII"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "all"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.append("..")
from datasets import Dataset, DatasetDict, load_dataset
from scipy.special import softmax
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

import wandb
from src.dataset.utils import (
    id2label,
    label2id,
    load_json_datasets,
    tokenize_and_align_labels,
)
from src.metrics import compute_metrics
from src.modeling import DebertaV2Baseline, DebertaV2FocalLoss

set_seed(42)

INFERENCE_MAX_LENGTH = 2300
MODEL_NAME = "microsoft/deberta-v3-base"
wandb_run_name = f"deberta-base-{INFERENCE_MAX_LENGTH}_focal_with_rewrited_mixtral"
model_save_path = f"/archive/ionov/pii/{wandb_run_name}"


if __name__ == "__main__":
    wandb.init("t-ionov")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df = load_json_datasets(
        [
            "data/essay/og_train_downsampled.json",
            # "data/essay/mixtral_train.json",
        ]
    )

    val_df = load_json_datasets(["data/essay/og_val.json"])
    dataset = DatasetDict(
        {"train": Dataset.from_pandas(train_df), "valid": Dataset.from_pandas(val_df)}
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, INFERENCE_MAX_LENGTH, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=16
    )

    model = DebertaV2Baseline.from_pretrained(
        # model = DebertaV2FocalLoss.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
    )
    model.to("cuda")

    BATCH_SIZE = 2
    TRAIN_STEPS = 10_000
    num_epochs = int(TRAIN_STEPS / (len(train_df) / BATCH_SIZE))
    print(num_epochs)
    training_args = TrainingArguments(
        output_dir="training_logs",
        learning_rate=1e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        warmup_steps=600,
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=1,
        metric_for_best_model="overall_f5_score",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=wandb_run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(4)]
    )

    train_loader = trainer.get_train_dataloader()

    trainer.train()
    wandb.finish()

    trainer.save_model(model_save_path)
