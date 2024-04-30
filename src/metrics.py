import evaluate
import numpy as np

seqeval_metrics = evaluate.load("seqeval")
from src.dataset.utils import id2label


def f5_score(precision, recall):
    return (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall + 1e-100)


def compute_metrics_from_labels(predictions, labels):
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metrics.compute(
        predictions=true_predictions, references=true_labels
    )
    for label, scores in results.items():
        if "overall" not in label:
            precision = scores["precision"]
            recall = scores["recall"]
            results[label]["f5_score"] = f5_score(precision, recall)
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    results["overall_f5_score"] = f5_score(precision, recall)

    return results


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return compute_metrics_from_labels(predictions, labels)
