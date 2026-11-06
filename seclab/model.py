"""Load the model and turn its next-token predictions into a phishing score."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import prompts


def load(name: str, device: str = "cpu"):
    """Load a chat model and its tokenizer.

    On a GPU the weights load in float16 to halve memory. On a CPU they
    stay in float32 because float16 math on a CPU is slow or unsupported.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def position_ids(mask: torch.Tensor) -> torch.Tensor:
    """Positions 0, 1, 2, ... counted from the first real token, so left padding does not shift them."""
    return (mask.cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def score(model, tokenizer, emails, system: str = prompts.SYSTEM, batch_size: int = 8) -> np.ndarray:
    """Return p(phishing) for each email.

    The model reads the prompt and predicts the next token. We keep only
    the logits for "Yes" and "No" and apply a softmax to those two:

        p(phishing) = exp(z_yes) / (exp(z_yes) + exp(z_no))

    This gives a probability for every email, so we can draw ROC curves,
    and we never have to parse free-form text.
    """
    yes_id, no_id = prompts.answer_ids(tokenizer)
    device = next(model.parameters()).device
    old_side = tokenizer.padding_side
    # Pad on the left so the last position in every row is the answer slot.
    tokenizer.padding_side = "left"
    out = []
    try:
        for i in range(0, len(emails), batch_size):
            batch = [prompts.prompt_text(tokenizer, e, system) for e in emails[i : i + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            # logits_to_keep=1 computes logits for the last position only.
            # Full logits would be batch x length x 151,936 numbers.
            logits = model(**enc, position_ids=position_ids(enc["attention_mask"]), logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1).float()
            out.append(torch.softmax(pair, dim=1)[:, 0].cpu().numpy())
    finally:
        tokenizer.padding_side = old_side
    return np.concatenate(out) if out else np.array([])
