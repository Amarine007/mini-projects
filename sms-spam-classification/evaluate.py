"""
Stage 5 of the pipeline: scoring the fitted classifier on the held-out test
split, with spam as the positive class.

Reports accuracy, precision, recall, F1 and ROC-AUC, the full
`classification_report`, and the confusion matrix -- calling out false
positives and false negatives separately, because the two costs are not
symmetric here: a legitimate message filed as spam is worse than a spam that
slips through. Also dumps the tokens that most favor spam, as a sanity check
that the model learned something sensible.

Run on its own against the saved artifacts; writes
`artifacts/test_metrics.json` for `plots.py` to draw from:

    python -m sms_spam_classification.evaluate
"""

import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from . import config
from .data import load_splits
from .features import load_vectorizer, transform_splits
from .model import load_model, predict, top_spam_indicators


def score_split(model, X, y, split_name):
    """Scores one split on accuracy/precision/recall/F1/ROC-AUC (spam = positive class)."""
    y_pred, y_proba = predict(model, X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    print(f"{split_name} -- " + ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items()))

    return metrics, y_pred, y_proba


def report(model, vectorizer, y_test, y_pred, metrics):
    """Prints the classification report, the confusion matrix, and the top spam tokens."""
    print("=" * 30)
    print("Classification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=config.DISPLAY_NAMES, digits=4))

    conf_mat = confusion_matrix(y_test, y_pred)
    _, false_pos, false_neg, _ = conf_mat.ravel()

    print("Confusion Matrix (Test):")
    print(conf_mat)
    print(f"Ham flagged as spam (false positives): {false_pos}")
    print(f"Spam that slipped through (false negatives): {false_neg}")
    print(f"Accuracy Score is: {metrics['accuracy'] * 100:.2f}%")

    print("=" * 30)
    print("Top spam-indicating tokens:")
    for token, score in top_spam_indicators(model, vectorizer):
        print(f"  {token:<20} {score:.3f}")

    return conf_mat


def save_metrics(metrics, conf_mat, y_test, y_proba):
    """Persists everything `plots.py` needs to redraw the report figure without refitting."""
    config.ensure_artifact_dirs()

    payload = {
        "metrics": {name: round(float(value), 6) for name, value in metrics.items()},
        "confusion_matrix": conf_mat.tolist(),
        "threshold": config.DECISION_THRESHOLD,
        "y_true": [int(value) for value in y_test],
        "y_proba": [round(float(value), 6) for value in y_proba],
    }

    config.METRICS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics -> {config.METRICS_PATH}")

    return payload


def run(model, vectorizer, matrices, trainval=None):
    """The whole stage: score train+val (if given) and test, report, persist metrics."""
    if trainval is not None:
        score_split(model, trainval[0], trainval[1], "Train+Val")

    X_test, y_test = matrices["test"]
    metrics, y_pred, y_proba = score_split(model, X_test, y_test, "Test")

    conf_mat = report(model, vectorizer, y_test, y_pred, metrics)

    return save_metrics(metrics, conf_mat, y_test, y_proba)


def main():
    splits = load_splits()
    vectorizer = load_vectorizer()
    model = load_model()
    matrices = transform_splits(vectorizer, splits)

    run(model, vectorizer, matrices)


if __name__ == "__main__":
    main()
