# NLP_ods_course

## Intoduction
<FIILME>

## Setup

- Python 3.10+
- 26+ GB GPU (one A100-SXM4-40GB used)
- wandb account
- kaggle account

- accept rules 

```
https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data
```

- load data
```
kaggle competitions download -c pii-detection-removal-from-educational-data
```

- clone repo
```
git clone https://github.com/sir-timio/NLP_ods_course
```


```
pip install -r requirements.txt
```
structure
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

- run fake generator
```
python src/generation/make_fake_pii.py
```
- run llm rewriting with PII

```
pybooks/llm_rewriting.ipynb 
or 
src/generation/llm_rewriting.py
```

- insert fake data into generated essays

```
pybooks/fill_ner.ipynb 
or 
src/generation/fill_ner.py
```

- configure and fit model

```
python train.py
```

logged results:

https://api.wandb.ai/links/ods_nlp_course/lihg45my