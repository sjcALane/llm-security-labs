"""Run settings shared by the notebooks.

Set the environment variable SECLAB_SMOKE=1 to shrink every size so a
notebook finishes in a few minutes on a CPU. CI uses this to check that
each lab runs from top to bottom.
"""

import os

import torch

SMOKE = os.environ.get("SECLAB_SMOKE", "0") == "1"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 260


def size(full: int, smoke: int) -> int:
    """Return the full size normally and the smoke size in smoke mode."""
    return smoke if SMOKE else full


def device() -> str:
    """Use the GPU when one is available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# Sizes used by the notebooks.
N_TRAIN = size(2000, 24)  # emails used for LoRA fine-tuning
N_TEST = size(500, 40)  # emails used for every evaluation
BATCH = 8
# Emails per forward pass during training. On a CPU we use 2 to save memory.
MICRO_BATCH = None if device() == "cuda" else 2
