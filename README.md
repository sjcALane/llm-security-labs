# LLM Security Labs

Fine-tune a small open language model to detect phishing email, attack it, defend it, and deploy it.

This is a six-lab series for an undergraduate computer security course. Each lab is a Colab notebook that runs on a free T4 GPU. Behind the notebooks is `seclab`, a small tested Python package, so the notebooks stay short and the code is reusable.

## The labs

| | Lab | What students do |
|---|---|---|
| 1 | [A classical baseline](notebooks/01_baseline.ipynb) | Clean the data, train TF-IDF with logistic regression, pick a threshold, find the shortcuts the model learned |
| 2 | [Zero-shot LLM and prompt injection](notebooks/02_zero_shot_and_prompt_injection.ipynb) | Turn next-token logits into a calibrated score, then break the model with instructions hidden in the email |
| 3 | [Fine-tuning with LoRA](notebooks/03_lora_finetuning.ipynb) | Derive the LoRA parameter count by hand, mask the loss to the answer tokens, train and evaluate an adapter |
| 4 | [Evasion attacks and defenses](notebooks/04_evasion_attacks_and_defenses.ipynb) | Homoglyph, zero-width, padding and injection attacks, then Unicode normalization and a hardened prompt |
| 5 | [Data poisoning](notebooks/05_data_poisoning.ipynb) | Plant a backdoor with 5% of the phishing training emails, show clean accuracy hides it, audit the data |
| 6 | [Deployment](notebooks/06_deployment.ipynb) | Merge the adapter, serve it with FastAPI behind security controls, measure latency and throughput, threat model it against the OWASP Top 10 for LLM Applications |

Each lab fits a 60 to 80 minute class period and ends with written questions.

## How it works

**One prompt, one token.** The model answers `Yes` or `No` to "Is this email a phishing attempt?" We read the two logits at the answer position and compute

$$p(\text{phishing}) = \sigma(z_{\text{Yes}} - z_{\text{No}})$$

so every email gets a probability, ROC curves work, and nothing depends on parsing generated text. Zero-shot scoring, fine-tuning and serving all use the same prompt.

**LoRA on the attention projections.** Rank 8 on `q_proj`, `k_proj`, `v_proj` and `o_proj` of Qwen2.5-0.5B-Instruct trains 1,081,344 parameters, 0.22% of the model.

**Loss on the answer only.** Prompt tokens are excluded from the loss. Batches are padded on the left so the answer always sits in the last positions, and the model computes logits only for those positions (`logits_to_keep`). With a vocabulary of about 150,000 tokens, full logits for a batch of eight 400-token prompts would take close to 2 GB in float32, so this cut is what lets training fit on small hardware.

**Duplicates removed before splitting.** About 1,100 emails in the raw data are exact copies. Left in, they leak between training and test sets.

## The package

```
seclab/
  data.py       download, clean, deduplicate, stratified split
  prompts.py    the shared prompt and answer tokens
  model.py      load a model, score emails from Yes/No logits
  lora.py       LoRA setup, answer-only loss, training loop with gradient accumulation
  attacks.py    homoglyph, zero-width, padding, prompt injection, backdoor poisoning
  defenses.py   Unicode normalization and look-alike mapping
  metrics.py    precision, recall, FPR, AUC, attack success rate
  serve.py      FastAPI service: size limit, normalization, rate limit, hashed logs
scripts/
  run_all.py        the whole series as one experiment, writes results/results.md
  run_notebooks.py  executes every notebook top to bottom
tests/              pytest suite, runs on a CPU in seconds
```

## Running it

In Colab, open any lab from the table and run the first cell. It installs the package from this repository.

On your own machine:

```bash
pip install -e ".[dev]"
pytest -q                                   # unit tests, CPU, a few seconds
SECLAB_SMOKE=1 python scripts/run_notebooks.py   # every notebook with tiny sizes
python scripts/run_all.py                   # full experiment, needs a GPU
```

`SECLAB_SMOKE=1` shrinks every size so a notebook finishes on a CPU. CI runs the tests on every push and can run all notebooks in smoke mode on demand.

## Data and model

- Emails: [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset), a copy of the Kaggle Phishing Email Detection dataset, LGPL-3.0. Downloaded at run time and not stored here.
- Model: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), Apache 2.0. Set `MODEL_NAME` in `seclab/config.py` to try another chat model.

## Author

Andrew Lane, Ph.D., Department of Mathematics and Computer Science, St. Joseph's University, New York.

Code is MIT licensed.
