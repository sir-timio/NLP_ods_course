import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import hydra
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    os.environ["WANDB_LOG_MODEL"] = str(cfg.wandb.log_model).lower()
    os.environ["WANDB_WATCH"] = cfg.wandb.watch

    set_seed(42)
    wandb.init(project=cfg.wandb.project)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    dataset_path = os.path.join(cfg.paths.dataset_path, "og_train_downsampled.json")
    ext_dataset_path = os.path.join(cfg.paths.dataset_path, "mixtral_train.json")
    train_df = load_json_datasets([dataset_path, ext_dataset_path])
    # train_df = load_json_datasets([dataset_path])
    val_df = load_json_datasets([os.path.join(cfg.paths.dataset_path, "og_val.json")])
    dataset = DatasetDict(
        {"train": Dataset.from_pandas(train_df), "valid": Dataset.from_pandas(val_df)}
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(
            x, cfg.model.inference_max_length, tokenizer
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=16
    )

    model = DebertaV2FocalLoss.from_pretrained(
        cfg.model.name,
        id2label=id2label,
        label2id=label2id,
    )
    model.to("cuda")

    num_epochs = int(cfg.model.train_steps / (len(train_df) / cfg.model.batch_size))
    wandb_run_name = (
        f"{cfg.model.name}-{cfg.model.inference_max_length}_{cfg.model.suffix}"
    )
    training_args = TrainingArguments(
        output_dir=cfg.trainer.output_dir,
        learning_rate=cfg.model.learning_rate,
        per_device_train_batch_size=cfg.model.batch_size,
        per_device_eval_batch_size=cfg.model.batch_size,
        num_train_epochs=num_epochs,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        weight_decay=cfg.model.weight_decay,
        evaluation_strategy=cfg.trainer.evaluation_strategy,
        save_strategy=cfg.trainer.save_strategy,
        logging_strategy=cfg.trainer.logging_strategy,
        warmup_steps=cfg.model.warmup_steps,
        eval_steps=cfg.model.eval_steps,
        save_steps=cfg.model.save_steps,
        logging_steps=cfg.model.logging_steps,
        save_total_limit=cfg.trainer.save_total_limit,
        metric_for_best_model=cfg.trainer.metric_for_best_model,
        greater_is_better=cfg.trainer.greater_is_better,
        load_best_model_at_end=cfg.trainer.load_best_model_at_end,
        report_to=cfg.trainer.report_to,
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
    )

    trainer.train()
    trainer.save_model("/archive/ionov/pii/focal_with_ext")
    wandb.finish()

    model_save_path = cfg.paths.model_save_path.format(wandb_run_name=wandb_run_name)
    trainer.save_model(model_save_path)


if __name__ == "__main__":
    main()
