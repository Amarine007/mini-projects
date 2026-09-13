"""
The trained pipeline used the way it would actually be used: raw message in,
0 (ham/real) or 1 (spam) out.

Walks the same path as training -- clean the text, transform with the saved
vectorizer, score with the saved model -- which is the point of keeping the
cleaner in its own module: there is no second, drifting copy of it here.

    python -m sms_spam_classification.predict
    python -m sms_spam_classification.predict "Claim your FREE prize now"
    cat messages.txt | python -m sms_spam_classification.predict -
"""

import sys

from . import config
from .features import load_vectorizer
from .model import load_model, predict as predict_batch
from .text_cleaning import clean_text


# used when no message is passed on the command line
DEMO_MESSAGES = [
    "Hey, are we still meeting for lunch tomorrow at noon?",
    "WINNER!! You have been selected to receive a FREE $1000 gift card. Claim now: http://bit.ly/claim",
]


def classify(messages, vectorizer=None, model=None):
    """Returns [(message, label, P(spam))] for a list of raw messages."""
    vectorizer = vectorizer if vectorizer is not None else load_vectorizer()
    model = model if model is not None else load_model()

    X = vectorizer.transform([clean_text(message) for message in messages])
    labels, probabilities = predict_batch(model, X)

    return list(zip(messages, labels, probabilities))


def read_messages(argv):
    """Messages come from argv, or from stdin when the only argument is `-`."""
    if argv == ["-"]:
        return [line.strip() for line in sys.stdin if line.strip()]

    # drop flags so a stray --no-plot from a pipeline-style invocation isn't classified
    messages = [arg for arg in argv if not arg.startswith("--")]

    return messages or DEMO_MESSAGES


def main():
    results = classify(read_messages(sys.argv[1:]))

    for message, label, probability in results:
        print(f"[{label}] {config.CLASS_NAMES[label]:<4} P(spam)={probability:.4f} -- {message[:70]}")


if __name__ == "__main__":
    main()
