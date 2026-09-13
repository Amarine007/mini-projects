"""
SMS spam filtering: ham (0, legitimate) vs spam (1).

    [Raw message] -> [Text cleaning] -> [TF-IDF vectorizer]
                  -> [Multinomial Naive Bayes] -> [Spam / Ham prediction]

Dataset: SMS Spam Collection (Kaggle: uciml/sms-spam-collection-dataset), 5,572
labeled English SMS messages, downloaded via `kagglehub`.

Model: Multinomial Naive Bayes over TF-IDF features -- fit in closed form, so
the per-epoch loop the other scripts in this project use is replaced by a sweep
over the Laplace smoothing strength `alpha`, tuned on the validation split.

Modules, in pipeline order (each is runnable on its own with `python -m`):

    config          shared constants and artifact paths
    text_cleaning   lowercase / placeholders / punctuation / stemming
    data            download, clean, investigate, 70/15/15 split
    features        fit TF-IDF on train, transform every split
    model           MultinomialNB construction, prediction, persistence (no CLI)
    train           alpha sweep on validation, refit the winner on train+val
    evaluate        precision / recall / F1 / ROC-AUC + report on the test split
    plots           class distribution, alpha sweep, confusion matrix, PR curve
    predict         raw message -> 0 (ham) or 1 (spam)
    pipeline        runs every stage back to back

See README.md for the run order and what each stage leaves in `artifacts/`.
"""
