"""
Stage 1 of the pipeline: [Raw message] -> [Text cleaning].

Pure text in, pure text out -- no dataset, no model, no disk. Lowercasing,
URL/email/number placeholders, punctuation stripping, whitespace tokenization,
and a small hand-rolled suffix-stripping stemmer (`simple_stem()`) so no extra
NLP dependency (nltk) is required. English stopwords are left alone here; the
TF-IDF vectorizer in `features.py` drops those.

Run on its own to eyeball what the cleaner does to a message:

    python -m sms_spam_classification.text_cleaning
    python -m sms_spam_classification.text_cleaning "WINNER!! Claim your FREE prize"
"""

import re
import string
import sys


# Precompiled patterns used by clean_text()
URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUMBER_RE = re.compile(r"\b\d[\d.,]*\b")
PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_RE = re.compile(r"\s+")

# Shown by the demo below when no message is passed on the command line
DEMO_MESSAGES = [
    "Hey, are we still meeting for lunch tomorrow at noon?",
    "WINNER!! You have been selected to receive a FREE $1000 gift card. Claim now: http://bit.ly/claim",
    "Txt STOP to 87121 to cancel. Charges apply -- reply-info@promo.co.uk",
]


def simple_stem(token):
    """A minimal suffix-stripping stemmer (Porter step-1 in spirit, no nltk needed)."""
    for suffix, replacement in (("sses", "ss"), ("ies", "y"), ("ied", "y")):
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)] + replacement

    for suffix in ("ingly", "edly", "ing", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]

    return token


def clean_text(text):
    """Lowercases, swaps urls/emails/numbers for placeholders, strips punctuation, stems."""
    text = str(text).lower()

    # collapse high-signal-but-always-unique tokens into placeholders so they generalize
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)

    text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    return " ".join(simple_stem(token) for token in text.split())


def clean_series(series):
    """Applies `clean_text` across a pandas Series of raw messages."""
    return series.apply(clean_text)


def main():
    messages = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or DEMO_MESSAGES

    for message in messages:
        print("raw:  ", message)
        print("clean:", clean_text(message))
        print("-" * 30)


if __name__ == "__main__":
    main()
