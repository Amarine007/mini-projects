"""
The wiring: runs every stage back to back in one process, in order.

    [Raw message] -> [Text cleaning] -> [TF-IDF vectorizer]
                  -> [Multinomial Naive Bayes] -> [Spam / Ham prediction]

    data -> features -> train -> evaluate -> plots -> predict

Stages hand objects to each other directly here (no reloading from disk between
them), while still writing their artifacts, so afterwards any single stage can
be re-run on its own against what the previous one left in `artifacts/`.

    python -m sms_spam_classification            # same thing, via __main__.py
    python -m sms_spam_classification.pipeline
    python -m sms_spam_classification.pipeline --no-plot
"""

import sys

from . import data
from . import features
from . import train
from . import evaluate
from . import plots
from . import predict


def run(show_plots=True):
    print("=" * 30)
    print("STAGE 1/5: data")
    _, splits = data.prepare(show_plot=show_plots)

    print("=" * 30)
    print("STAGE 2/5: features (TF-IDF)")
    vectorizer, matrices = features.build(splits)

    print("=" * 30)
    print("STAGE 3/5: train (Multinomial Naive Bayes)")
    model, sweep, trainval = train.run(matrices)

    print("=" * 30)
    print("STAGE 4/5: evaluate")
    metrics_payload = evaluate.run(model, vectorizer, matrices, trainval=trainval)

    print("=" * 30)
    print("STAGE 5/5: plots")
    if show_plots:
        plots.plot_evaluation(sweep, metrics_payload)
    else:
        print("Skipped (--no-plot).")

    # a couple of ad-hoc predictions to sanity check the end-to-end path
    print("=" * 30)
    print("Sample predictions:")
    for message, label, probability in predict.classify(predict.DEMO_MESSAGES, vectorizer, model):
        print(f"[{label}] P(spam)={probability:.4f} -- {message[:70]}")

    return model, metrics_payload


def main():
    run(show_plots="--no-plot" not in sys.argv)


if __name__ == "__main__":
    main()
