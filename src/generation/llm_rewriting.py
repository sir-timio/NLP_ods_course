# it is strongly recommended to use the appropriate file in pybooks
import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from hydra import compose, initialize
from IPython.core.display import HTML, display
from omegaconf import OmegaConf
from spacy import displacy
from spacy.lang.en import English
from spacy.tokens import Doc, Span

import wandb


def setup_cli():
    parser = argparse.ArgumentParser(
        description="Data generation and entity management with CLI support."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="../conf/",
        help="Directory containing Hydra configuration files.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="generation_conf",
        help="Name of the configuration file to load.",
    )
    return parser


def load_hydra_config(config_dir, config_name):
    initialize(version_base=None, config_path=config_dir, job_name="rewriting")
    cfg = compose(config_name=config_name)
    return cfg


def initialize_engine(cfg) -> LLMEngine:
    engine_args = EngineArgs(**cfg.engine)
    return LLMEngine.from_engine_args(engine_args)


PII_ENTS = [
    ("name", "NAME_STUDENT", "James Brown"),
    ("email", "EMAIL", "example@email.com"),
    ("personal_url", "URL_PERSONAL", "https://example.com"),
    ("username", "USERNAME", "john42"),
    ("address", "STREET_ADDRESS", "221B, Baker Street, London"),
    ("phone_num", "PHONE_NUM", "+1 212 555 0188"),
    ("userid", "ID_NUM", "123456789"),
]

ENT_COMBINATIONS = [
    *[(ent,) for ent in PII_ENTS],
    *[(PII_ENTS[0], ent) for ent in PII_ENTS],
    *[comb for comb in combinations(PII_ENTS[:4], 3)],
]

def sample_ent_combination():
    return np.random.choice(ENT_COMBINATIONS, p=None)

def dict2str(d):
    return "\n".join([f"{k}={v}" for k, v in d.items()])

def build_request(prompt_format, ent_combination, essay=None):
    ents_to_generate = {
        ent_type: [ent_text]
        for ent_description, ent_type, ent_text in ent_combination
    }
    pii_str = "\n".join(
        [
            f"{ent_description}={ent_text}"
            for ent_description, ent_type, ent_text in ent_combination
        ]
    )
    prompt = (
        prompt_format.format(pii_str, "")
        if essay is None
        else prompt_format.format(essay, pii_str)
    )

    request = {
        "prompt": prompt,
        "ents_to_generate": ents_to_generate,
    }
    return request

def create_requests(essays, prompt_format) -> list[dict]:
    essays = [None] if essays is None else essays
    generation_requests = []
    for essay in essays:
    ent_combination = sample_ent_combination()
    request = build_request(prompt_format, ent_combination, essay)
    generation_requests.append(request)

    return generation_requests
    
def process_requests(engine, generation_requests):
    """Continuously pro cess a list of prompts and handle the outputs."""

    generated_examples = []
    for request_id, request_data in enumerate(generation_requests):
        prompt = request_data["prompt"]
        sampling_params = SamplingParams(**request_data["sampling_params"])
        lora_request = (
            None
            if not request_data["lora_params"]
            else LoRARequest(**request_data["lora_params"])
        )

        engine.add_request(str(request_id), prompt, sampling_params, lora_request)

    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                for output in request_output.outputs:
                    generated_text = output.text
                    request_data = generation_requests[int(request_output.request_id)]
                    generated_examples.append(
                        {"generated_text": generated_text, **request_data}
                    )
    return generated_examples


def main():
    parser = setup_cli()
    args = parser.parse_args()

    cfg = load_hydra_config(args.config_dir, args.config_name)

    # Initialize WandB with parameters from the Hydra config
    wandb_cfg = cfg.wandb
    os.environ["WANDB_PROJECT"] = wandb_cfg.project
    os.environ["WANDB_ENTITY"] = wandb_cfg.entity
    os.environ["WANDB_JOB_TYPE"] = wandb_cfg.job_type
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Load prompt format from file
    with open(cfg.prompt_path, "r") as file:
        prompt_format = file.read()

    # Read essays for rewriting
    orig_essays_df = pd.read_json(cfg.original_essays_path)
    generation_requests = create_requests(essays, prompt_format)
    
    essays_with_pii = orig_essays_df[orig_essays_df.labels.apply(lambda x: set(["O", "B-NAME_STUDENT"]).issubset(set(x)))]
    orig_essays_df =  orig_essays_df[orig_essays_df.labels.apply(lambda x: set(x) == set(["O"]))]
    essays = orig_essays_df["full_text"].tolist()[:cfg.n_samples]
    ENT_COMBINATIONS = np.array(ENT_COMBINATIONS, dtype="object")
    wandb.login()
    with wandb.init(name=wandb_cfg.run_name + "2", job_type=wandb_cfg.job_type) as run:
        # Assuming log_df is generated above
        table = wandb.Table(dataframe=log_df)
        run.summary["generation_examples_table"] = table


if __name__ == "__main__":
    main()
