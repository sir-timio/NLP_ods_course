from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import hydra
from omegaconf import OmegaConf, ListConfig
import numpy as np
import pandas as pd
import wandb
import os

from typing import Optional, List, Tuple

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

# from baseline.utils import tokenize_with_spacy

def init_wandb(cfg):
    wandb_cfg = cfg.wandb
    os.environ["WANDB_PROJECT"] = wandb_cfg.project
    os.environ["WANDB_ENTITY"] = wandb_cfg.entity
    os.environ["WANDB_JOB_TYPE"] = wandb_cfg.job_type
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "all"

    os.environ["HYDRA_FULL_ERROR"] = "1"


def create_test_prompts(cfg) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.
    
    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """

    # orig_essays = df = pd.read_json(cfg.original_essays_path)


    with open(cfg.prompt_path, "r") as file:
        prompt = file.read()
    print("____________________________________________________________________________________________________________")
    print("Prompt: ")
    print(prompt)
    print("\n\n\n\n\n")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.engine.model)


    chat = [
        {"role": "user", "content": prompt}
    ]

    prompt_with_chat_template = tokenizer.apply_chat_template(chat, tokenize=False)

    return [
        (
            prompt_with_chat_template, 
            SamplingParams(**cfg.sampling_params),
            None # LoRARequest("pii-lora-adapter", 1, cfg.lora_path)
        )
    ]

    return [
        (
            prompt, 
            SamplingParams(**cfg.sampling_params),
            None # LoRARequest("pii-lora-adapter", 1, cfg.lora_path)
        )
        for prompt in cfg.prompts
    ]


def initialize_engine(cfg) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.

    engine_args = EngineArgs(**cfg.engine)
    return LLMEngine.from_engine_args(engine_args)


def process_requests(engine,
                     test_prompts,
                     cfg):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    generated_examples = []
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                for output in request_output.outputs:
                    generated_text = output.text
                    generated_examples.append({**cfg.sampling_params, **{"prompt": prompt, "response": generated_text}})
    return generated_examples
    
def log_prompts(generated_examples, cfg):
    df = pd.DataFrame().from_records(generated_examples)
    df.to_json(cfg.output_path, orient="records", indent=4)

    for text in df["response"].tolist():
        print("###########################################################################################")
        print("###########################################################################################")
        print("###########################################################################################")

        print(text)
        print("\n\n")

    # with wandb.init(name=cfg.wandb.run_name, job_type=cfg.wandb.job_type) as run:
    #     table = wandb.Table(dataframe=df)
    #     run.summary["generation_examples_table"] = table


@hydra.main(version_base=None, config_path="../conf/generation", config_name="generation_conf")
def main(cfg):
    init_wandb(cfg)
    engine = initialize_engine(cfg)    
    test_prompts = create_test_prompts(cfg)
    generated_examples = process_requests(engine, test_prompts, cfg)
    log_prompts(generated_examples, cfg)

if __name__ == "__main__":
    main()
