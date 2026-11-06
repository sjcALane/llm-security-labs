"""Run the whole lab series as one experiment and write results/results.md.

    python scripts/run_all.py                 # full sizes, needs a GPU
    python scripts/run_all.py --train 400 --test 200 --micro-batch 2   # small CPU run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from seclab import attacks, config, data, defenses, lora, metrics, model, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--poison-frac", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    dev = config.device()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()

    df = data.load_emails()
    train_full, _, test_full = data.split(df, seed=config.SEED)
    train = data.sample(train_full, args.train, seed=config.SEED)
    test = data.sample(test_full, args.test, seed=config.SEED)
    y = test["label"].to_numpy()
    emails = test["text"].tolist()
    rows, asr = {}, {}

    # Lab 1: classical baseline on the full training split.
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train_full["text"]), train_full["label"])
    rows["TF-IDF + logistic regression"] = metrics.evaluate(y, clf.predict_proba(vec.transform(emails))[:, 1])

    # Lab 2: zero-shot.
    base, tok = model.load(config.MODEL_NAME, dev)
    p_zero = model.score(base, tok, emails)
    rows["Zero-shot LLM"] = metrics.evaluate(y, p_zero)
    phish0 = [attacks.inject(e) for e, label in zip(emails, y) if label == 1]
    asr["Zero-shot LLM, prompt injection"] = metrics.attack_success_rate(
        np.ones(len(phish0), dtype=int), p_zero[y == 1], model.score(base, tok, phish0)
    )

    # Lab 3: LoRA fine-tuning.
    ft = lora.add_lora(base)
    lora.train(ft, tok, train["text"].tolist(), train["label"].tolist(),
               batch_size=args.batch, micro_batch=args.micro_batch, log_every=25)
    p_ft = model.score(ft, tok, emails)
    rows[f"LoRA fine-tuned ({args.train} emails)"] = metrics.evaluate(y, p_ft)

    # Lab 4: evasion attacks and the normalization defense.
    # Attacks only matter on phishing emails, so we score only those.
    phish = [e for e, label in zip(emails, y) if label == 1]
    yp = np.ones(len(phish), dtype=int)
    pp = p_ft[y == 1]
    attacked = {
        "homoglyph": [attacks.homoglyph(e, seed=i) for i, e in enumerate(phish)],
        "zero-width": [attacks.zero_width(e) for e in phish],
        "benign padding": [attacks.pad_benign(e, seed=i) for i, e in enumerate(phish)],
        "prompt injection": [attacks.inject(e) for e in phish],
    }
    for name, texts in attacked.items():
        p_att = model.score(ft, tok, texts)
        p_def = model.score(ft, tok, [defenses.normalize(t) for t in texts])
        asr[f"Fine-tuned, {name}"] = metrics.attack_success_rate(yp, pp, p_att)
        asr[f"Fine-tuned, {name}, normalized"] = metrics.attack_success_rate(yp, pp, p_def)
    asr["Fine-tuned, prompt injection, hardened prompt"] = metrics.attack_success_rate(
        yp, model.score(ft, tok, phish, system=prompts.SYSTEM_HARDENED),
        model.score(ft, tok, attacked["prompt injection"], system=prompts.SYSTEM_HARDENED),
    )
    triggered = [attacks.add_trigger(e) for e in phish]
    asr["Clean fine-tuned model, trigger phrase"] = metrics.attack_success_rate(
        yp, pp, model.score(ft, tok, triggered)
    )

    # Lab 5: backdoor poisoning. Retrain from a fresh base model.
    del ft, base
    base, tok = model.load(config.MODEL_NAME, dev)
    poisoned, idx = attacks.poison(train, args.poison_frac, seed=config.SEED)
    bd = lora.add_lora(base)
    lora.train(bd, tok, poisoned["text"].tolist(), poisoned["label"].tolist(),
               batch_size=args.batch, micro_batch=args.micro_batch, log_every=25)
    p_bd = model.score(bd, tok, emails)
    rows[f"Backdoored ({len(idx)} poisoned emails)"] = metrics.evaluate(y, p_bd)
    asr["Backdoored model, trigger phrase"] = metrics.attack_success_rate(
        yp, p_bd[y == 1], model.score(bd, tok, triggered)
    )

    minutes = (time.time() - t0) / 60
    hw = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    lines = [
        "# Results",
        "",
        f"Model: {config.MODEL_NAME}. Hardware: {hw}. Train emails: {args.train}. "
        f"Test emails: {args.test}. Run time: {minutes:.0f} min.",
        "",
        "## Detection",
        "",
        metrics.table(rows).to_markdown(),
        "",
        "## Attack success rate",
        "",
        "Share of phishing emails the model caught that the attack let through.",
        "",
        "| Setting | Attack success rate |",
        "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in asr.items()],
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps({"detection": rows, "asr": asr, "args": vars(args)}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
