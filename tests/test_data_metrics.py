import numpy as np
import pandas as pd
import pytest

from seclab import data, metrics


def raw():
    rows = [
        ("Please verify your account now at http://bad.example", "Phishing Email"),
        ("Please verify your account now at http://bad.example", "Phishing Email"),
        ("Meeting moved to Friday, see agenda attached for details", "Safe Email"),
        ("empty", "Safe Email"),
        (None, "Safe Email"),
        ("short", "Phishing Email"),
    ]
    rows += [(f"routine safe message number {i} about the project plan", "Safe Email") for i in range(20)]
    rows += [(f"urgent: your password expires, click link {i} to keep access", "Phishing Email") for i in range(20)]
    return pd.DataFrame(rows, columns=["Email Text", "Email Type"])


def test_clean_drops_bad_rows_and_duplicates():
    df = data.clean(raw(), max_chars=30)
    assert list(df.columns) == ["text", "label"]
    assert not df["text"].str.lower().eq("empty").any()
    assert df["text"].str.len().max() <= 30
    assert len(df) == 1 + 1 + 20 + 20  # one copy of the duplicate, short row gone
    assert set(df["label"]) == {0, 1}


def test_split_is_disjoint_and_stratified():
    df = data.clean(raw())
    train, val, test = data.split(df, test_frac=0.25, val_frac=0.25, seed=1)
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train["text"]) & set(test["text"])
    assert not set(train["text"]) & set(val["text"])
    for part in (train, val, test):
        assert abs(part["label"].mean() - df["label"].mean()) < 0.15


def test_sample_size():
    df = data.clean(raw())
    assert len(data.sample(df, 10)) == 10
    assert len(data.sample(df, 10_000)) == len(df)


def test_evaluate_known_confusion():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]
    m = metrics.evaluate(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 5)


def test_attack_success_rate_counts_only_caught_phishing():
    y = np.array([1, 1, 1, 0])
    clean = np.array([0.9, 0.9, 0.1, 0.1])  # third phishing email was already missed
    attacked = np.array([0.2, 0.9, 0.1, 0.9])
    assert metrics.attack_success_rate(y, clean, attacked) == pytest.approx(0.5)


def test_table_skips_non_metric_entries():
    m = metrics.evaluate([1, 0, 1, 0], [0.9, 0.1, 0.4, 0.6])
    t = metrics.table({"model": m, "some ASR": 0.5, "ASR by attack": {"homoglyph": 0.2}})
    assert list(t.index) == ["model"]
