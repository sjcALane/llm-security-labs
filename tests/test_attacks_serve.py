import pandas as pd
from fastapi.testclient import TestClient

from seclab import attacks, defenses
from seclab.serve import create_app

EMAIL = "your account is suspended, open this page and type your password"


def test_homoglyph_looks_different_to_code_and_normalize_undoes_it():
    bad = attacks.homoglyph(EMAIL, rate=1.0)
    assert bad != EMAIL
    assert len(bad) == len(EMAIL)
    assert defenses.normalize(bad) == EMAIL


def test_zero_width_is_invisible_and_removed():
    bad = attacks.zero_width(EMAIL, every=2)
    assert attacks.ZERO_WIDTH_SPACE in bad
    assert bad.replace(attacks.ZERO_WIDTH_SPACE, "") == EMAIL
    assert defenses.normalize(bad) == EMAIL
    assert defenses.suspicious_chars(bad) == bad.count(attacks.ZERO_WIDTH_SPACE)


def test_inject_and_pad_append():
    assert attacks.inject(EMAIL).startswith(EMAIL)
    assert attacks.pad_benign(EMAIL, n=2).startswith(EMAIL)


def test_poison_flips_the_right_rows():
    df = pd.DataFrame({"text": [f"phish {i}" for i in range(50)] + [f"safe {i}" for i in range(50)],
                       "label": [1] * 50 + [0] * 50})
    poisoned, idx = attacks.poison(df, frac=0.1, seed=3)
    assert len(idx) == 5
    assert (poisoned.loc[idx, "label"] == 0).all()
    assert poisoned.loc[idx, "text"].str.endswith(attacks.TRIGGER).all()
    assert (df["label"] == [1] * 50 + [0] * 50).all()  # original untouched


def client(rate_limit=100):
    seen = []

    def fake_score(texts):
        seen.extend(texts)
        return [0.9 if "password" in t else 0.1 for t in texts]

    return TestClient(create_app(fake_score, model_version="test", rate_limit=rate_limit)), seen


def test_classify_normalizes_before_scoring():
    c, seen = client()
    r = c.post("/classify", json={"text": attacks.homoglyph(EMAIL, rate=1.0)})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "phishing"
    assert body["suspicious_chars"] > 0
    assert seen[-1] == EMAIL


def test_rejects_oversized_input():
    c, _ = client()
    assert c.post("/classify", json={"text": "a" * 5001}).status_code == 422


def test_rate_limit():
    c, _ = client(rate_limit=3)
    codes = [c.post("/classify", json={"text": "hello there"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
