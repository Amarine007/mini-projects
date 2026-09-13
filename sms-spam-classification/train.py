"""
Stage 4 of the pipeline: fitting the Naive Bayes classifier.

Naive Bayes has no epochs, so the per-epoch loop the other scripts in this
project use is replaced by a sweep over the Laplace smoothing strength `alpha`,
scored on the validation split. The best `alpha` is then refit on train + val
(more data for the final model) before the test split is touched at all in
`evaluate.py`.

Run on its own from the saved splits and vectorizer; writes
`artifacts/naive_bayes.joblib` and `artifacts/alpha_sweep.json`:

    python -m sms_spam_classification.train
"""

import json

import numpy as np
from scipy.sparse import vstack
from sklearn.metrics import f1_score

from . import config
from .data import load_splits
from .features import load_vectorizer, transform_splits
from .model import build_model, predict, save_model


def sweep_alphas(matrices):
    """Fits one model per alpha, scoring spam F1 on train and validation."""
    sweep = {"alphas": [], "train_f1": [], "val_f1": []}
    best_alpha = config.ALPHAS[0]
    best_val_f1 = -1.0

    X_train, y_train = matrices["train"]
    X_val, y_val = matrices["val"]

    for alpha in config.ALPHAS:
        model = build_model(alpha)
        model.fit(X_train, y_train)

        train_f1 = f1_score(y_train, predict(model, X_train)[0], zero_division=0)
        val_f1 = f1_score(y_val, predict(model, X_val)[0], zero_division=0)

        sweep["alphas"].append(alpha)
        sweep["train_f1"].append(round(train_f1, 4))
        sweep["val_f1"].append(round(val_f1, 4))

        print(f"alpha: {alpha:<5} Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_alpha = alpha

    print("=" * 30)
    print(f"Best alpha: {best_alpha} (Validation F1: {best_val_f1:.4f})")

    sweep["best_alpha"] = best_alpha
    sweep["best_val_f1"] = round(best_val_f1, 4)

    return sweep


def fit_final_model(matrices, alpha):
    """Refits the chosen alpha on train + validation, stacking the two feature matrices."""
    X_train, y_train = matrices["train"]
    X_val, y_val = matrices["val"]

    X_trainval = vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    model = build_model(alpha)
    model.fit(X_trainval, y_trainval)

    return model, (X_trainval, y_trainval)


def save_sweep(sweep):
    config.ensure_artifact_dirs()
    config.SWEEP_PATH.write_text(json.dumps(sweep, indent=2))
    print(f"Saved sweep history -> {config.SWEEP_PATH}")


def run(matrices):
    """The whole stage: sweep alpha, refit the winner on train+val, save both outputs."""
    sweep = sweep_alphas(matrices)
    model, trainval = fit_final_model(matrices, sweep["best_alpha"])

    save_model(model)
    save_sweep(sweep)

    return model, sweep, trainval


def main():
    splits = load_splits()
    vectorizer = load_vectorizer()
    matrices = transform_splits(vectorizer, splits)

    run(matrices)


if __name__ == "__main__":
    main()
