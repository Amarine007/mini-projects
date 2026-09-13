"""
Every figure the pipeline draws, kept in one place so the stages that produce
numbers stay free of matplotlib.

`data.py` calls `plot_class_distribution()`; the final report figure is built
from the two json artifacts (`alpha_sweep.json` from train, `test_metrics.json`
from evaluate), which is what lets this module run on its own -- re-draw the
plots after a finished run without refitting anything:

    python -m sms_spam_classification.plots
"""

import json

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from . import config


def _finish(fig, save_path):
    """Saves a figure next to the other artifacts and shows it (a no-op when headless)."""
    config.ensure_artifact_dirs()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"Saved plot -> {save_path}")
    plt.show()


def plot_class_distribution(data_df):
    """Plots the ham/spam balance and the message-length distribution per class."""
    class_counts = data_df["Label"].value_counts()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))

    axs[0].pie(class_counts, labels=class_counts.index, autopct="%1.1f%%")
    axs[0].set_title("Class Distribution (ham vs spam)")

    for label in class_counts.index:
        lengths = data_df.loc[data_df["Label"] == label, "Message"].str.len()
        axs[1].hist(lengths, bins=50, alpha=0.6, label=label)
    axs[1].set_title("Message Length by Class")
    axs[1].set_xlabel("Characters")
    axs[1].set_ylabel("Messages")
    axs[1].legend()

    _finish(fig, config.PLOTS_DIR / "class_distribution.png")


def _draw_alpha_sweep(ax, sweep):
    """The smoothing sweep -- Naive Bayes' stand-in for a per-epoch learning curve."""
    ax.plot(sweep["alphas"], sweep["train_f1"], marker="o", label="Training F1")
    ax.plot(sweep["alphas"], sweep["val_f1"], marker="o", label="Validation F1")
    ax.set_title("F1 vs Smoothing Strength (alpha)")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("F1 (spam)")
    ax.set_xscale("log")
    ax.legend()


def _draw_test_metrics(ax, metrics):
    """Headline test metrics as a labeled bar chart."""
    names = list(metrics.keys())
    values = [metrics[name] * 100 for name in names]

    bars = ax.bar(names, values, color="tab:blue")
    ax.bar_label(bars, fmt="%.2f")
    ax.set_title("Test Metrics (spam = positive class)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim([0, 105])


def _draw_confusion_matrix(ax, conf_mat):
    ax.imshow(conf_mat, cmap="Blues")
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(2), config.DISPLAY_NAMES)
    ax.set_yticks(range(2), config.DISPLAY_NAMES)

    largest = max(max(row) for row in conf_mat)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, conf_mat[i][j], ha="center", va="center",
                    color="white" if conf_mat[i][j] > largest / 2 else "black")


def _draw_precision_recall(ax, y_true, y_proba):
    """The precision/recall trade-off across every possible decision threshold."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)

    ax.plot(recalls, precisions)
    ax.set_title("Precision-Recall Curve (Test)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])


def plot_evaluation(sweep, metrics_payload):
    """The 2x2 report figure: alpha sweep, test metrics, confusion matrix, PR curve."""
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    _draw_alpha_sweep(axs[0, 0], sweep)
    _draw_test_metrics(axs[0, 1], metrics_payload["metrics"])
    _draw_confusion_matrix(axs[1, 0], metrics_payload["confusion_matrix"])
    _draw_precision_recall(axs[1, 1], metrics_payload["y_true"], metrics_payload["y_proba"])

    _finish(fig, config.PLOTS_DIR / "evaluation.png")


def main():
    for path in (config.SWEEP_PATH, config.METRICS_PATH):
        if not path.exists():
            raise SystemExit(f"Missing {path.name}. Run the train and evaluate stages first.")

    sweep = json.loads(config.SWEEP_PATH.read_text())
    metrics_payload = json.loads(config.METRICS_PATH.read_text())

    plot_evaluation(sweep, metrics_payload)


if __name__ == "__main__":
    main()
