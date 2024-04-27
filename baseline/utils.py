import spacy
from spacy.tokens import Span, Doc
from spacy import displacy
from spacy.lang.en import English

import numpy as np


# nlp = spacy.blank("en")
# nlp = spacy.load("en_core_web_sm")
word_tokenizer = English().tokenizer

options = {
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


def tokenize_with_spacy(text):
    tokenized_text = word_tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return tokens, trailing_whitespace

def visualize_ents(tokens, trailing_whitespace, ents):
    doc = Doc(word_tokenizer.vocab, words=tokens, spaces=trailing_whitespace, ents=ents)
    html = displacy.render(doc, style="ent", jupyter=False, options=options)
    return html

def apply_threshold(preds, threshold, O_label_id):
    preds_without_O = preds[:,:,:O_label_id]
    O_preds = preds[:,:,O_label_id]
    preds_final = np.where(O_preds < threshold, preds_without_O.argmax(-1), preds.argmax(-1))
    return preds_final
