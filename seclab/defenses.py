"""Input normalization: undo character tricks before the model sees the text."""

from __future__ import annotations

import unicodedata

from .attacks import HOMOGLYPHS

# Characters that take up no space on screen.
INVISIBLE = {
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "‎",  # left-to-right mark
    "‏",  # right-to-left mark
    "⁠",  # word joiner
    "﻿",  # byte order mark
}

# Cyrillic and Greek look-alikes mapped back to Latin.
CONFUSABLES = {v: k for k, v in HOMOGLYPHS.items()}
CONFUSABLES.update({"ο": "o", "α": "a", "ε": "e", "ρ": "p"})


def normalize(text: str) -> str:
    """NFKC normalization, then drop invisible characters, then map look-alikes to Latin.

    NFKC folds compatibility characters such as full-width letters into
    their plain forms. It does not touch Cyrillic letters because those
    are real letters in another alphabet, so we map them ourselves.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in INVISIBLE)
    return "".join(CONFUSABLES.get(ch, ch) for ch in text)


def suspicious_chars(text: str) -> int:
    """Count characters that normalize() would remove or change."""
    return sum(1 for ch in text if ch in INVISIBLE or ch in CONFUSABLES)
