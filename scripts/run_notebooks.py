"""Execute every lab notebook from top to bottom.

    SECLAB_SMOKE=1 python scripts/run_notebooks.py           # quick check on a CPU
    python scripts/run_notebooks.py --save executed/        # keep the outputs

Notebooks run in order inside one working folder, so later labs can load
the adapter and results saved by earlier ones.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="notebook file names (default: all)")
    ap.add_argument("--save", help="folder for executed copies")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    args = ap.parse_args()

    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if args.names:
        paths = [p for p in paths if p.name in args.names]
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as work:
        for path in paths:
            start = time.time()
            nb = nbformat.read(path, as_version=4)
            NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": work}}).execute()
            print(f"ok  {path.name}  {time.time() - start:.0f}s", flush=True)
            if save:
                nbformat.write(nb, save / path.name)


if __name__ == "__main__":
    main()
