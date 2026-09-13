"""
The model itself: [TF-IDF features] -> [Multinomial Naive Bayes] -> [Spam / Ham].

`MultinomialNB` is the classic generative model for word-count / TF-IDF
features. It is fit in closed form from token statistics, so there is no epoch
loop anywhere in this package -- the only knob is the Laplace smoothing
strength `alpha`, which `train.py` sweeps.

Definitions only (construct, predict, persist, inspect); no stage logic and
nothing to run here.
"""

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from . import config


def build_model(alpha):
    return MultinomialNB(alpha=alpha)


def predict(model, X, threshold=config.DECISION_THRESHOLD):
    """Returns (hard labels, P(spam)) using an explicit threshold on the spam probability."""
    spam_proba = model.predict_proba(X)[:, 1]
    return (spam_proba >= threshold).astype(int), spam_proba


def top_spam_indicators(model, vectorizer, top_n=15):
    """Lists the tokens whose log-probability most favors spam over ham."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]  # spam minus ham
    order = np.argsort(log_ratio)[::-1][:top_n]

    return list(zip(feature_names[order], log_ratio[order]))


def save_model(model):
    config.ensure_artifact_dirs()
    joblib.dump(model, config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")


def load_model():
    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Missing {config.MODEL_PATH.name}. "
                         "Run `python -m sms_spam_classification.train` first.")

    return joblib.load(config.MODEL_PATH)
