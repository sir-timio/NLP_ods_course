from collections import defaultdict

import numpy as np
from spacy import displacy
from spacy.lang.en import English
from spacy.tokens import Doc

ENT_REPLACE_ORDER = [
    "URL_PERSONAL",
    "EMAIL",
    "USERNAME",
    "STREET_ADDRESS",
    "NAME_STUDENT",
    "PHONE_NUM",
    "ID_NUM",
]

UNIQUE_CLASS_LABELS = set(ENT_REPLACE_ORDER)

UNIQUE_ENT_TAGS = [
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-EMAIL",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
    "I-USERNAME",
    "O",
    "B-NAME_STUDENT_NON_PII",
    "I-NAME_STUDENT_NON_PII",
    "B-URL_PERSONAL_NON_PII",
]

ID_2_LABEL = {i: label for i, label in enumerate(UNIQUE_ENT_TAGS)}
LABEL_2_ID = {v: k for k, v in ID_2_LABEL.items()}
O_LABEL_ID = LABEL_2_ID["O"]


LABEL2ENT_SPECIAL_TOKEN = {l: l + "_TOKEN" for l in UNIQUE_CLASS_LABELS}
ENT_SPECIAL_TOKEN2LABEL = {l + "_TOKEN": l for l in UNIQUE_CLASS_LABELS}
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


def remove_bio(tag):
    return tag if tag == "O" else tag[2:]


def gen_ent_dict(row):
    true_ents_dict = defaultdict(set)
    labels = row["labels"]
    n = len(labels)
    ent_start_token_id = 0
    while ent_start_token_id < n:
        if labels[ent_start_token_id] != "O":
            ent_end_token_id = ent_start_token_id + 1
            while ent_end_token_id < n and remove_bio(
                labels[ent_start_token_id]
            ) == remove_bio(labels[ent_end_token_id]):
                ent_end_token_id += 1

            ent_text = "".join(
                [
                    t + (" " if w else "")
                    for t, w in zip(
                        row["tokens"][ent_start_token_id:ent_end_token_id],
                        row["trailing_whitespace"][ent_start_token_id:ent_end_token_id],
                    )
                ]
            )
            ent_text = ent_text.rstrip()
            ent_label_class = remove_bio(labels[ent_start_token_id])

            true_ents_dict[ent_label_class].add(ent_text)

            ent_start_token_id = ent_end_token_id
        else:
            ent_start_token_id += 1

    row["true_ents_dict"] = {k: list(v) for k, v in true_ents_dict.items()}
    return row


def tokenize_with_spacy(text, tokenizer=WORD_TOKENIZER):
    tokenized_text = tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return tokens, trailing_whitespace


def replace_ents_with_labels(row):
    text = row["text"]
    row["ents_present_in_text"] = defaultdict(list)
    for ent_label in ENT_REPLACE_ORDER:
        if ent_label not in row["true_ents_dict"]:
            continue

        for ent_text in row["true_ents_dict"][ent_label]:
            row["ents_present_in_text"][ent_label].append(ent_text in row["text"])
            if ent_text in text:
                text = text.replace(ent_text, LABEL2ENT_SPECIAL_TOKEN[ent_label])

    row["text_with_ent_holders"] = text
    return row


def tokenize_df_with_spacy(row):
    tokens, trailing_whitespace = tokenize_with_spacy(row["text_with_ent_holders"])
    row["tokens"] = tokens
    row["trailing_whitespace"] = trailing_whitespace
    row["labels"] = ["O"] * len(tokens)
    return row


def mark_ent_label_tokens(row):
    row["label2position"] = defaultdict(list)  # {x: [] for x in UNIQUE_CLASS_LABELS}
    tokens = row["tokens"]
    for i, tok in enumerate(tokens):
        if tok in ENTITY_SPECIAL_TOKENS:
            enity_label = ENT_SPECIAL_TOKEN2LABEL[tok]
            row["labels"][i] = "B-" + enity_label
            row["label2position"][enity_label].append(i)
    return row


def replace_labels_with_ents(row):
    label2position = row["label2position"]
    ents_dict = row["label2ent"]

    ent_mentions = [
        (ent_label, ent_text, pos)
        for ent_label, ent_text in ents_dict.items()
        for pos in label2position[ent_label]
    ]
    sorted_ent_mentions = sorted(ent_mentions, key=lambda x: x[-1], reverse=True)

    for ent_label, ent_text, pos in sorted_ent_mentions:
        assert len(ent_text) == 1
        ent_text = ent_text[0]
        ent_tokens, ent_trailing_whitespace = tokenize_with_spacy(ent_text)
        ent_bio_tags = ["B-" + ent_label] + ["I-" + ent_label] * (len(ent_tokens) - 1)

        assert len(ent_tokens) == len(ent_trailing_whitespace) == len(ent_bio_tags)

        for k, v in [
            ("tokens", ent_tokens),
            ("trailing_whitespace", ent_trailing_whitespace),
            ("labels", ent_bio_tags),
        ]:
            row[k].pop(pos)
            row[k][pos:pos] = v

    return row


def add_ner_tags(row):
    row["ner_tags"] = [LABEL_2_ID[x] for x in row["labels"]]
    return row


def visualize_ents(tokens, trailing_whitespace, ents):
    doc = Doc(WORD_TOKENIZER.vocab, words=tokens, spaces=trailing_whitespace, ents=ents)
    html = displacy.render(
        doc, style="ent", jupyter=False, options=VISUALIZATION_OPTIONS
    )
    return html


def tokens2text(tokens, trailing_whitespace):
    return "".join(
        [
            token + (" " if whitespace else "")
            for token, whitespace in zip(tokens, trailing_whitespace)
        ]
    )


def apply_threshold(preds, threshold, O_label_id):
    preds_without_O = preds[:, :, :O_label_id]
    O_preds = preds[:, :, O_label_id]
    preds_final = np.where(
        O_preds < threshold, preds_without_O.argmax(-1), preds.argmax(-1)
    )
    return preds_final
