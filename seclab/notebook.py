"""Small helpers used at the top of each lab notebook."""

from __future__ import annotations

import os
from pathlib import Path

from . import config


def workdir() -> Path:
    """Folder for saved adapters and results.

    In Colab with Google Drive mounted this is MyDrive/seclab, so files
    survive between sessions. Otherwise it is ./seclab_runs.
    """
    drive = Path("/content/drive/MyDrive")
    root = drive / "seclab" if drive.exists() else Path("seclab_runs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def banner():
    dev = config.device()
    mode = "SMOKE TEST (tiny sizes)" if config.SMOKE else "full"
    print(f"device: {dev}   mode: {mode}   model: {config.MODEL_NAME}")
    if dev == "cpu" and not config.SMOKE:
        print("No GPU found. In Colab use Runtime > Change runtime type > T4 GPU.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def save_result(name: str, value) -> None:
    """Add one result to results.json so later labs can compare against it."""
    import json

    path = workdir() / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results[name] = value
    path.write_text(json.dumps(results, indent=2))


def load_results() -> dict:
    import json

    path = workdir() / "results.json"
    return json.loads(path.read_text()) if path.exists() else {}


def free_memory() -> None:
    """Release memory held by models you deleted with del."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
