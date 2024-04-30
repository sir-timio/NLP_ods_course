# it is strongly recommended to use the appropriate file in pybooks
# python fill_ner.py --texts_path "path/to/texts.json" --entity_path "path/to/entities.csv"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from utils import (
    UNIQUE_LABELS,
    mark_ent_label_tokens,
    replace_ents_with_labels,
    replace_labels_with_ents,
    tokenize_df_with_spacy,
)


def load_data(text_path, entity_path):
    with open(text_path, "r") as file:
        data = json.load(file)
    texts_df = pd.DataFrame().from_records(data)
    ents_df = pd.read_csv(entity_path).drop(columns="COUNTRY")
    return texts_df, ents_df


def create_entity_combinations(df):
    return (
        df.applymap(lambda x: [x])
        .agg(lambda row: row.to_dict(), axis=1)
        .reset_index(drop=True)
    )


def create_label_dictionary(df, label_to_entity):
    def add_label_dict(row):
        row["true_ents_dict"] = {
            label: [label_to_entity[label]]
            for label in UNIQUE_LABELS
            if row[label] is not None
        }
        return row

    return df.apply(add_label_dict, axis=1)


def assign_random_entities(text_df, ents_comb_df):
    n_unique = len(ents_comb_df)
    n_ents = len(text_df)
    rand_indexes = np.random.randint(n_unique, size=n_ents)
    text_df["label2ent"] = pd.Series(ents_comb_df.iloc[rand_indexes].to_numpy())
    return text_df


def process_texts(df):
    if "label2position" in df.columns:
        df = df.apply(replace_labels_with_ents, axis=1)
    else:
        df = (
            df.apply(replace_ents_with_labels, axis=1)
            .apply(tokenize_df_with_spacy, axis=1)
            .apply(mark_ent_label_tokens, axis=1)
            .apply(replace_labels_with_ents, axis=1)
        )
    return df


def filter_entities(df):
    df["has_ents"] = df["labels"].apply(lambda labels: len(set(labels)) > 3)
    return df[df["has_ents"]]


def save_data(df, path):
    df.to_json(path)


def setup_cli_parser():
    parser = argparse.ArgumentParser(
        description="Process and handle entity replacements in texts."
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="../data/essay/rewrited_train.json",
        help="Path to the text data file (default: ../data/essay/rewrited_train.json)",
    )
    parser.add_argument(
        "--entity_path",
        type=str,
        default="../data/faker_pii.csv",
        help="Path to the entity data file (default: ../data/faker_pii.csv)",
    )
    return parser


def main():
    parser = setup_cli_parser()
    args = parser.parse_args()

    texts_df, ents_df = load_data(args.texts_path, args.entity_path)
    ents_comb_df = create_entity_combinations(ents_df)

    pii_ents = [
        ("name", "NAME_STUDENT", "James Brown"),
        ("email", "EMAIL", "example@email.com"),
        ("personal_url", "URL_PERSONAL", "https://example.com"),
        ("username", "USERNAME", "john42"),
        ("address", "STREET_ADDRESS", "221B, Baker Street, London"),
        ("phone_num", "PHONE_NUM", "+1 212 555 0188"),
        ("userid", "ID_NUM", "123456789"),
    ]
    label_to_entity = {l: e for _, l, e in pii_ents}

    texts_df = create_label_dictionary(texts_df, label_to_entity)
    texts_df = assign_random_entities(texts_df, ents_comb_df)

    texts_df = process_texts(texts_df)
    texts_df = filter_entities(texts_df)

    save_path = args.texts_path.replace(".json", "_processed.json")
    save_data(texts_df, save_path)


if __name__ == "__main__":
    main()
