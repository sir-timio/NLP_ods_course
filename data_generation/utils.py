import wandb
import json
from pathlib import Path
import pandas as pd

import spacy
from spacy.tokens import Span, Doc
from spacy import displacy
from spacy.lang.en import English


ENT_REPLACE_ORDER = [
    "URL_PERSONAL",
    "EMAIL",
    "USERNAME",
    "STREET_ADDRESS",
    "NAME_STUDENT",
    "PHONE_NUM",
    "ID_NUM",
]

UNIQUE_LABELS = {
    "URL_PERSONAL",
    "EMAIL",
    "USERNAME",
    "STREET_ADDRESS",
    "NAME_STUDENT",
    "PHONE_NUM",
    "ID_NUM"
}

LABEL2ENT_SPECIAL_TOKEN = {l : l + "_TOKEN" for l in UNIQUE_LABELS}
ENT_SPECIAL_TOKEN2LABEL = {l + "_TOKEN": l for l in UNIQUE_LABELS}
ENTITY_SPECIAL_TOKENS = set(LABEL2ENT_SPECIAL_TOKEN.values())
WORD_TOKENIZER = English().tokenizer

VISUALIZATION_OPTIONS = {
        "colors": {
            "B-NAME_STUDENT": "aqua",
            "I-NAME_STUDENT": "skyblue",
            "B-EMAIL": "limegreen",
            "I-EMAIL": "lime",
            "B-USERNAME": "hotpink",
            "I-USERNAME": "lightpink",
            "B-ID_NUM": "purple",
            "I-ID_NUM": "rebeccapurple",
            "B-PHONE_NUM": "red",
            "I-PHONE_NUM": "salmon",
            "B-URL_PERSONAL": "silver",
            "I-URL_PERSONAL": "lightgray",
            "B-STREET_ADDRESS": "brown",
            "I-STREET_ADDRESS": "chocolate",
        }
    }

def tokenize_with_spacy(text, tokenizer=WORD_TOKENIZER):
    tokenized_text = tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return tokens, trailing_whitespace

def replace_ents_with_labels(row):
    text = row["generated_text"]
    ents_present_in_generated_text = {}
    for ent_label in ENT_REPLACE_ORDER:
        if ent_label not in row["true_ents_dict"]:
            continue

        ent = row["true_ents_dict"][ent_label]
        assert len(ent) == 1

        ent_text = ent[0]
        ents_present_in_generated_text[ent_label] = ent_text in text 
        if ent_text in text :
            text = text.replace(ent_text, LABEL2ENT_SPECIAL_TOKEN[ent_label])

    row["generated_text_with_ent_labels"] = text
    row["ents_present_in_generated_text"] = ents_present_in_generated_text
    return row

def tokenize_df_with_spacy(row):
    tokens, trailing_whitespace = tokenize_with_spacy(row["generated_text_with_ent_labels"])
    row["tokens"] = tokens
    row["trailing_whitespace"] = trailing_whitespace
    row["labels"] = ["O"] * len(tokens)
    return row

def mark_ent_label_tokens(row):
    label2position = {x: [] for x in UNIQUE_LABELS}
    tokens = row["tokens"]
    for i, tok in enumerate(tokens):
        if tok in ENTITY_SPECIAL_TOKENS:
            enity_label = ENT_SPECIAL_TOKEN2LABEL[tok]
            row["labels"][i] = 'B-' + enity_label
            label2position[enity_label].append(i)
    row["label2position"] = label2position
    return row

def replace_labels_with_ents(row):
    label2position = row["label2position"]
    ents_dict = row["label2ent"]

    entity_mentions = [(ent_label, ent_text, pos) for ent_label, ent_text in ents_dict.items() for pos in label2position[ent_label]]
    sorted_entity_mentions = sorted(entity_mentions, key=lambda x: x[-1], reverse=True)

    for ent_label, ent_text, pos in sorted_entity_mentions:
        assert len(ent_text) == 1
        ent_text = ent_text[0]
        ent_tokens, ent_trailing_whitespace = tokenize_with_spacy(ent_text)
        ent_bio_tags = ["B-" + ent_label] + ["I-" + ent_label] * (len(ent_tokens) - 1)

        assert len(ent_tokens) == len(ent_trailing_whitespace) == len(ent_bio_tags)
            
        for k, v in [("tokens", ent_tokens), ("trailing_whitespace", ent_trailing_whitespace), ("labels", ent_bio_tags)]:
            row[k].pop(pos)
            row[k][pos:pos] = v

    return row


def visualize_ents(tokens, trailing_whitespace, ents):
    doc = Doc(WORD_TOKENIZER.vocab, words=tokens, spaces=trailing_whitespace, ents=ents)
    html = displacy.render(doc, style="ent", jupyter=False, options=VISUALIZATION_OPTIONS)
    return html
