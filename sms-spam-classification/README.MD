# SMS Spam Classification

TF-IDF + Multinomial Naive Bayes on the SMS Spam Collection (5,572 English messages; ham = 0, spam = 1).

```
raw message -> clean_text() -> TF-IDF -> MultinomialNB -> ham / spam
```

**Test results:** 98.19% accuracy · precision 0.988 · recall 0.863 · ROC-AUC 0.990
(1 false positive and 13 missed spam out of 775 messages).

## Quick start

Run from the project root (the folder above this one):

```
python -m sms_spam_classification                                   # full pipeline (--no-plot to skip figures)
python -m sms_spam_classification.predict "Are we still on for lunch?"
cat messages.txt | python -m sms_spam_classification.predict -      # one message per line
```

The dataset (Kaggle `uciml/sms-spam-collection-dataset`) is downloaded by `kagglehub` on first run, then cached.

## Stages

Each stage also runs on its own as `python -m sms_spam_classification.<stage>`. It reads whatever the
previous stage saved to `artifacts/`, so run them in order the first time. Artifacts can be regenerated,
so it's safe to delete them.

| Stage      | What it does                                         | Writes to `artifacts/`                   |
| ---------- | ---------------------------------------------------- | ---------------------------------------- |
| `data`     | Download, clean, encode labels, 70/15/15 split       | `processed.csv`, `train/val/test.csv`    |
| `features` | Fit TF-IDF on **train only**, transform every split  | `tfidf_vectorizer.joblib`                |
| `train`    | Sweep `alpha` by validation F1, refit on train + val | `naive_bayes.joblib`, `alpha_sweep.json` |
| `evaluate` | Test metrics, classification report, confusion matrix | `test_metrics.json`                      |
| `plots`    | Redraw figures from the two json files               | `plots/*.png`                            |

Other modules: `config.py` (all constants, hyperparameters, and paths), `text_cleaning.py` (also runnable
on a string), `model.py`, `predict.py`, and `pipeline.py` (runs every stage in one process).

## Notes

- **No epoch loop.** Naive Bayes is fit in closed form, so tuning means sweeping the smoothing strength
  `alpha` rather than training over epochs.
- **Cleaning** lowercases the text, replaces URLs, emails, and numbers with placeholder tokens, strips
  punctuation, and applies a hand-rolled stemmer, so there's no nltk dependency. The vectorizer drops
  stopwords.
- **Threshold.** Filing a real message as spam costs more than letting a spam through. Recall is held
  back by class imbalance (653 spam vs. 4,514 ham). To catch more spam at the cost of more false
  positives, lower `DECISION_THRESHOLD` in `config.py` (default 0.5).
