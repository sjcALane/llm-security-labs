"""Model tests on a tiny randomly initialized Qwen2 network.

Only the tokenizer is downloaded (a few MB), so these run quickly on a CPU.
"""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from seclab import lora, model, prompts
from seclab.config import MODEL_NAME


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture()
def tiny(tok):
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_answer_tokens_are_single(tok):
    yes, no = prompts.answer_ids(tok)
    assert yes != no


def test_encode_masks_the_prompt(tok):
    ids, labels = lora.encode_example(tok, "click here to verify", 1)
    graded = [t for t, l in zip(ids, labels) if l != -100]
    assert tok.decode(graded) == prompts.YES + tok.eos_token
    assert len(ids) == len(labels)


def test_score_returns_probabilities(tiny, tok):
    p = model.score(tiny, tok, ["hello", "verify your password now", "lunch at noon"], batch_size=2)
    assert p.shape == (3,)
    assert np.all((p >= 0) & (p <= 1))


def test_score_ignores_padding(tiny, tok):
    emails = ["short", "a much longer email that forces padding on the other row " * 3]
    together = model.score(tiny, tok, emails, batch_size=2)
    alone = np.array([model.score(tiny, tok, [e])[0] for e in emails])
    assert np.allclose(together, alone, atol=1e-4)


def test_lora_param_formula_matches_peft(tiny):
    peft_model = lora.add_lora(tiny, r=4, targets=("q_proj",))
    trainable, _ = lora.count_params(peft_model)
    q = tiny.config.hidden_size
    per_layer = lora.lora_param_count(q, q, 4)
    assert trainable == per_layer * tiny.config.num_hidden_layers


def test_training_lowers_loss(tiny, tok):
    peft_model = lora.add_lora(tiny, r=8)
    texts = ["verify your password at this link", "notes from today's meeting"] * 8
    labels = [1, 0] * 8
    losses = lora.train(peft_model, tok, texts, labels, epochs=4, batch_size=4, lr=5e-3, log_every=0)
    assert np.mean(losses[-4:]) < np.mean(losses[:4])


def test_micro_batches_match_full_batch(tok):
    """Gradient accumulation gives the same first-step loss as one big batch."""
    texts = ["verify your password at this link", "notes from today's meeting", "win a prize now", "lunch?"]
    labels = [1, 0, 1, 0]
    runs = []
    for mb in (4, 1):
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
        m = lora.add_lora(Qwen2ForCausalLM(cfg), r=4, dropout=0.0)
        runs.append(lora.train(m, tok, texts, labels, epochs=2, batch_size=4, micro_batch=mb, log_every=0))
    assert np.allclose(runs[0], runs[1], atol=1e-4)
