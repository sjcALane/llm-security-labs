"""LoRA fine-tuning.

LoRA freezes the pretrained weight matrix W (size d_out x d_in) and
learns a low-rank update instead:

    W' = W + (alpha / r) * B A,   B is d_out x r,  A is r x d_in

Only A and B are trained. For one matrix that is r * (d_in + d_out)
parameters instead of d_in * d_out.
"""

from __future__ import annotations

import math
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_linear_schedule_with_warmup

from . import prompts
from .model import position_ids

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    """Trainable parameters LoRA adds to one d_out x d_in matrix."""
    return r * (d_in + d_out)


def count_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def add_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05, targets=TARGETS):
    """Freeze the model and attach LoRA adapters to the attention projections."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def encode_example(tokenizer, email: str, label: int, system: str = prompts.SYSTEM, max_len: int = 512):
    """Token ids and training labels for one email.

    The target is the answer token followed by the end-of-turn token.
    Prompt positions get label -100, which tells PyTorch to skip them in
    the loss. The model is graded only on its answer.
    """
    prompt = prompts.prompt_text(tokenizer, email, system)
    answer = (prompts.YES if label == 1 else prompts.NO) + tokenizer.eos_token
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    p_ids = p_ids[: max_len - len(a_ids)]
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    return ids, labels


def _collate(batch, pad_id: int):
    """Pad on the left so every answer sits in the last positions of its row."""
    width = max(len(ids) for ids, _ in batch)
    input_ids, mask = [], []
    for ids, _ in batch:
        pad = width - len(ids)
        input_ids.append([pad_id] * pad + ids)
        mask.append([0] * pad + [1] * len(ids))
    return torch.tensor(input_ids), torch.tensor(mask)


def answer_loss(model, ids: torch.Tensor, mask: torch.Tensor, n_answer: int) -> torch.Tensor:
    """Cross-entropy on the answer tokens only.

    The token at position t is predicted by the logits at position t - 1.
    The answer fills the last n_answer positions, so we need logits for
    the last n_answer + 1 positions. Asking for only those saves a lot of
    memory, since each position has one logit per vocabulary word.
    """
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=position_ids(mask),
        logits_to_keep=n_answer + 1,
    )
    logits = out.logits[:, :-1, :].float()
    targets = ids[:, -n_answer:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def train(
    model,
    tokenizer,
    texts,
    labels,
    epochs: int = 1,
    batch_size: int = 8,
    micro_batch: int | None = None,
    lr: float = 2e-4,
    warmup_frac: float = 0.05,
    max_len: int = 512,
    system: str = prompts.SYSTEM,
    seed: int = 260,
    log_every: int = 10,
):
    """Train the LoRA adapters with AdamW and a linear learning-rate schedule.

    batch_size is the number of emails per optimizer step. If that does
    not fit in memory, set micro_batch smaller. The gradients of several
    micro-batches are added up before each step (gradient accumulation),
    which gives the same update with less memory.

    Returns the list of training losses, one per step.
    """
    micro_batch = micro_batch or batch_size
    device = next(model.parameters()).device
    examples = [encode_example(tokenizer, t, int(y), system, max_len) for t, y in zip(texts, labels)]
    n_answer = {sum(1 for l in lab if l != -100) for _, lab in examples}
    if len(n_answer) != 1:
        raise ValueError("Every answer must have the same number of tokens.")
    n_answer = n_answer.pop()
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = steps_per_epoch * epochs

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(warmup_frac * total_steps)), total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    losses, step, start = [], 0, time.time()
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=gen).tolist()
        for i in range(0, len(order), batch_size):
            batch = [examples[j] for j in order[i : i + batch_size]]
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for k in range(0, len(batch), micro_batch):
                part = batch[k : k + micro_batch]
                ids, mask = (x.to(device) for x in _collate(part, tokenizer.pad_token_id))
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = answer_loss(model, ids, mask, n_answer) * len(part) / len(batch)
                scaler.scale(loss).backward()
                step_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(step_loss)
            step += 1
            if log_every and (step % log_every == 0 or step == total_steps):
                print(f"step {step}/{total_steps}  loss {step_loss:.4f}  {time.time() - start:.0f}s")
    model.eval()
    return losses


def load_adapter(base_model, path: str):
    """Attach a saved LoRA adapter to a freshly loaded base model."""
    model = PeftModel.from_pretrained(base_model, path)
    model.eval()
    return model


def load_or_train(base_model, tokenizer, train_df, path: str, **train_kwargs):
    """Load the adapter saved at path. If there is none, train one and save it.

    Later labs call this so they work whether or not you kept the adapter
    from the LoRA lab.
    """
    from pathlib import Path

    if (Path(path) / "adapter_config.json").exists():
        print(f"Loading adapter from {path}")
        return load_adapter(base_model, path)
    print(f"No adapter at {path}. Training one now.")
    model = add_lora(base_model)
    train(model, tokenizer, train_df["text"].tolist(), train_df["label"].tolist(), **train_kwargs)
    model.save_pretrained(path)
    return model
