"""
Shared configuration for the SMS spam pipeline: dataset coordinates,
hyperparameters, and the on-disk artifact paths each stage reads and writes.

Every other module imports its constants from here so a stage run on its own
sees exactly the same settings as the same stage run inside the full pipeline.
"""

from pathlib import Path


# --- Dataset ---
DATASET_HANDLE = "uciml/sms-spam-collection-dataset"
CSV_NAME = "spam.csv"
CSV_ENCODING = "latin-1"   # the Kaggle csv is not utf-8

# --- Splitting ---
TRAIN_FRAC = 0.7           # remainder is split evenly into validation / test
RANDOM_STATE = 7

# --- Feature extraction (TF-IDF) ---
MAX_FEATURES = 5000        # vocabulary cap
NGRAM_RANGE = (1, 2)       # unigrams + bigrams ("free entry", "call now", ...)
MIN_DF = 2                 # ignore tokens appearing in a single message only

# --- Model (Multinomial Naive Bayes) ---
ALPHAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]  # Laplace smoothing values to sweep
DECISION_THRESHOLD = 0.5   # P(spam) at or above this is labeled spam

# --- Labels ---
CLASS_NAMES = ["ham", "spam"]          # index == encoded target: ham -> 0, spam -> 1
DISPLAY_NAMES = ["ham (0)", "spam (1)"]

# --- Artifact paths (everything a stage hands to the next stage) ---
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

PROCESSED_CSV = ARTIFACT_DIR / "processed.csv"       # cleaned + encoded full dataset
SPLIT_CSVS = {
    "train": ARTIFACT_DIR / "train.csv",
    "val": ARTIFACT_DIR / "val.csv",
    "test": ARTIFACT_DIR / "test.csv",
}
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACT_DIR / "naive_bayes.joblib"
SWEEP_PATH = ARTIFACT_DIR / "alpha_sweep.json"       # written by train, read by plots
METRICS_PATH = ARTIFACT_DIR / "test_metrics.json"    # written by evaluate, read by plots
PLOTS_DIR = ARTIFACT_DIR / "plots"


def ensure_artifact_dirs():
    """Creates the artifact folders on demand so no stage has to care about ordering."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
