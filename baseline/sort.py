import copy
import gc
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from spacy.lang.en import English
from torch import nn
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.models.deberta_v2 import (
    DebertaV2ForTokenClassification,
    DebertaV2TokenizerFast,
)
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

import wandb
