"""
Stage 2 of the pipeline: dataset acquisition, cleaning, investigation, and the
70/15/15 train/val/test split.

Downloads the SMS Spam Collection (Kaggle: uciml/sms-spam-collection-dataset)
via `kagglehub` -- non-interactive, and cached locally after the first run --
keeps the label/message columns, runs every message through
`text_cleaning.clean_text()`, and encodes the label as 0 = ham/real, 1 = spam
with a `LabelEncoder` fitted once on the full class list so all three splits
share one mapping.

Run on its own to fetch the data and write the split csvs into `artifacts/`
(add `--no-plot` to skip the class-distribution figure):

    python -m sms_spam_classification.data
"""

import os
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config
from . import plots
from .text_cleaning import clean_series


# fitted once on the full class list so every split shares the mapping: ham -> 0, spam -> 1
label_encoder = LabelEncoder().fit(config.CLASS_NAMES)


def download_dataset():
    """Fetches (or reuses the cached copy of) the dataset and returns its local folder."""
    import kagglehub
    return kagglehub.dataset_download(config.DATASET_HANDLE)


def load_and_preprocess_data(dataset_dir):
    """Reads the csv, keeps label/message, cleans the text, and encodes the target."""
    data_df = pd.read_csv(os.path.join(dataset_dir, config.CSV_NAME), encoding=config.CSV_ENCODING)

    # the csv ships three mostly-empty trailing columns; keep only label (v1) and message (v2)
    data_df = data_df[["v1", "v2"]].rename(columns={"v1": "Label", "v2": "Message"})
    data_df = data_df.dropna().drop_duplicates(subset="Message").reset_index(drop=True)

    data_df["CleanMessage"] = clean_series(data_df["Message"])
    data_df = data_df[data_df["CleanMessage"].str.len() > 0].reset_index(drop=True)

    data_df["Target"] = label_encoder.transform(data_df["Label"])

    return data_df


def investigate(data_df):
    """Prints the shape/head/class balance and an example of the cleaning step."""
    print("Data Shape is:", data_df.shape)
    print(data_df.head())

    print("Classes Distribution is:")
    print(data_df["Label"].value_counts())

    print("Label mapping:", dict(zip(config.CLASS_NAMES, label_encoder.transform(config.CLASS_NAMES))))

    print("Example cleaned message:")
    print("  raw:  ", data_df["Message"].iloc[0])
    print("  clean:", data_df["CleanMessage"].iloc[0])


def split_data(data_df):
    """70% train, then an even split of the remainder into validation and test."""
    train_df = data_df.sample(frac=config.TRAIN_FRAC, random_state=config.RANDOM_STATE)
    remainder_df = data_df.drop(train_df.index)

    val_df = remainder_df.sample(frac=0.5, random_state=config.RANDOM_STATE)
    test_df = remainder_df.drop(val_df.index)

    print("Training Shape:", train_df.shape)
    print("Validation Shape:", val_df.shape)
    print("Testing Shape:", test_df.shape)

    return {"train": train_df, "val": val_df, "test": test_df}


def save_splits(data_df, splits):
    """Writes the processed dataset and each split to `artifacts/` for later stages."""
    config.ensure_artifact_dirs()

    data_df.to_csv(config.PROCESSED_CSV, index=False)
    for name, split_df in splits.items():
        split_df.to_csv(config.SPLIT_CSVS[name], index=False)

    print(f"Saved processed data and splits -> {config.ARTIFACT_DIR}")


def load_splits():
    """Reads the split csvs back; used by the stages that run standalone."""
    missing = [path.name for path in config.SPLIT_CSVS.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Run `python -m sms_spam_classification.data` first.")

    return {name: pd.read_csv(path) for name, path in config.SPLIT_CSVS.items()}


def prepare(show_plot=True):
    """The whole stage: download -> clean -> investigate -> split -> save."""
    dataset_dir = download_dataset()
    data_df = load_and_preprocess_data(dataset_dir)

    investigate(data_df)
    if show_plot:
        plots.plot_class_distribution(data_df)

    splits = split_data(data_df)
    save_splits(data_df, splits)

    return data_df, splits


def main():
    prepare(show_plot="--no-plot" not in sys.argv)


if __name__ == "__main__":
    main()
