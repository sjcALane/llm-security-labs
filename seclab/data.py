"""Load and split the phishing email dataset.

Source: zefang-liu/phishing-email-dataset on Hugging Face, a copy of the
Kaggle "Phishing Email Detection" dataset (LGPL-3.0). The data is
downloaded at run time and is not stored in this repository.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ID = "zefang-liu/phishing-email-dataset"
FILENAME = "Phishing_Email.csv"

PHISHING = 1
SAFE = 0


def clean(df: pd.DataFrame, max_chars: int = 1000, min_chars: int = 20) -> pd.DataFrame:
    """Turn the raw CSV into two columns: text and label.

    Steps:
      1. Drop rows with no text and rows whose text is the word "empty".
      2. Drop very short emails.
      3. Drop exact duplicates. Without this step the same email can land
         in both the training and test sets, which inflates test scores.
      4. Keep the first max_chars characters of each email.
    """
    out = pd.DataFrame(
        {
            "text": df["Email Text"].astype("string").str.strip(),
            "label": (df["Email Type"] == "Phishing Email").astype(int),
        }
    )
    out = out.dropna(subset=["text"])
    out = out[out["text"].str.lower() != "empty"]
    out = out[out["text"].str.len() >= min_chars]
    out = out.drop_duplicates(subset="text")
    out["text"] = out["text"].str.slice(0, max_chars).astype(str)
    return out.reset_index(drop=True)


def load_emails(max_chars: int = 1000) -> pd.DataFrame:
    """Download the dataset (cached after the first call) and clean it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset")
    return clean(pd.read_csv(path), max_chars=max_chars)


def split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 260):
    """Stratified train / validation / test split.

    Stratified means each part keeps the same phishing-to-safe ratio
    as the full dataset.
    """
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    rel_val = val_frac / (1 - test_frac)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def sample(df: pd.DataFrame, n: int, seed: int = 260) -> pd.DataFrame:
    """Stratified sample of n rows (or all rows if n is larger)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    part, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return part.reset_index(drop=True)
