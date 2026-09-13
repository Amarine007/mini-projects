"""
Stage 3 of the pipeline: [Cleaned text] -> [TF-IDF vectorizer].

The vectorizer is fitted on the training split only -- fitting it on everything
would leak val/test vocabulary and idf weights into training -- then reused to
transform the other splits and, later, any message handed to `predict.py`.

Run on its own to fit the vectorizer from the saved splits and write
`artifacts/tfidf_vectorizer.joblib`:

    python -m sms_spam_classification.features
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .data import load_splits


def build_vectorizer():
    return TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        stop_words="english",
        sublinear_tf=True,   # 1 + log(tf) damps repeated tokens in long messages
    )


def fit_vectorizer(train_texts):
    """Fits TF-IDF on the training text and reports the resulting vocabulary size."""
    vectorizer = build_vectorizer()
    vectorizer.fit(train_texts)

    print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))

    return vectorizer


def transform_splits(vectorizer, splits):
    """Returns {split: (X, y)} with the fitted vectorizer applied to each split."""
    matrices = {}
    for name, split_df in splits.items():
        X = vectorizer.transform(split_df["CleanMessage"])
        y = split_df["Target"].to_numpy()
        matrices[name] = (X, y)

    print("Training matrix shape:", matrices["train"][0].shape)

    return matrices


def save_vectorizer(vectorizer):
    config.ensure_artifact_dirs()
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print(f"Saved vectorizer -> {config.VECTORIZER_PATH}")


def load_vectorizer():
    if not config.VECTORIZER_PATH.exists():
        raise SystemExit(f"Missing {config.VECTORIZER_PATH.name}. "
                         "Run `python -m sms_spam_classification.features` first.")

    return joblib.load(config.VECTORIZER_PATH)


def build(splits):
    """The whole stage: fit on train, transform every split, save the vectorizer."""
    vectorizer = fit_vectorizer(splits["train"]["CleanMessage"])
    matrices = transform_splits(vectorizer, splits)
    save_vectorizer(vectorizer)

    return vectorizer, matrices


def main():
    build(load_splits())


if __name__ == "__main__":
    main()
