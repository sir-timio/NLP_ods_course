# Personally Identifiable Information Data Detection. NLP Course Project
Anna Marshalova, Olga Tikhobaeva, Timur Ionov

# Prerequisites

- Python 3.10+
- 26+ GB GPU (we used one A100-SXM4-40GB)
- W&B account
- kaggle account

# Setup

1. Accept rules of the kaggle comprtition to access the dataset

```
https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data
```

2. Load data
```
kaggle competitions download -c pii-detection-removal-from-educational-data
```

3. Clone repo
```
git clone https://github.com/sir-timio/NLP_ods_course
```
4. Install dependencies
```
pip install -r requirements.txt
```
# Repo structure
```
.
├── conf
│   ├── generation_conf.yaml
│   └── prompts
│       └── rewriting_prompt_v1.txt
├── data
│   ├── essay
│   │   ├── mixtral_train.json
│   │   ├── og_train_downsampled.json
│   │   ├── og_train.json
│   │   ├── og_val.json
│   │   ├── orig_train.json
│   └── faker_pii.csv
├── pybooks
│   ├── dataset_logging.ipynb
│   ├── eda.ipynb
│   ├── fill_ner.ipynb
│   └── llm_rewriting.ipynb
├── README.md
├── src
│   ├── dataset
│   │   └── utils.py
│   ├── generation
│   │   ├── fill_ner.py
│   │   ├── llm_rewriting.py
│   │   ├── make_fake_pii.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── modeling
│   │   ├── deberta_base.py
│   │   ├── deberta_focal.py
│   │   ├── __init__.py
```

# Usage

1. Run fake PII generator
```
python src/generation/make_fake_pii.py
```
2. Run LLM text rewriting with PII

```
pybooks/llm_rewriting.ipynb 
or 
src/generation/llm_rewriting.py
```

3. Insert fake data into the generated essays

```
pybooks/fill_ner.ipynb 
or 
src/generation/fill_ner.py
```

4. Configure and fit model

```
python train.py
```
# Results
All metrics are logged in this [W&B report](https://api.wandb.ai/links/ods_nlp_course/lihg45my).

