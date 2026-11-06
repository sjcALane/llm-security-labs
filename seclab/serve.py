"""A small web service around the detector.

The service takes the detector as a function, so the same code runs with
the real model in the lab and with a stand-in function in the tests.

Security controls:
  * input size limit (rejects oversized requests)
  * normalization before scoring (defenses.normalize)
  * per-client rate limit
  * logs a hash of each email, never the email itself
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict, deque
from typing import Callable, Sequence

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .defenses import normalize, suspicious_chars

log = logging.getLogger("seclab.serve")

ScoreFn = Callable[[Sequence[str]], Sequence[float]]


class EmailIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class Verdict(BaseModel):
    label: str
    p_phishing: float
    suspicious_chars: int
    model_version: str
    latency_ms: float


class RateLimiter:
    """Allow at most `limit` requests per client in any `window` seconds."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        q = self.hits[client]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def create_app(
    score_fn: ScoreFn,
    model_version: str = "dev",
    threshold: float = 0.5,
    rate_limit: int = 60,
) -> FastAPI:
    app = FastAPI(title="Phishing detector")
    limiter = RateLimiter(rate_limit)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": model_version}

    @app.post("/classify", response_model=Verdict)
    def classify(email: EmailIn, request: Request):
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="Too many requests")
        start = time.perf_counter()
        flagged = suspicious_chars(email.text)
        p = float(score_fn([normalize(email.text)])[0])
        ms = (time.perf_counter() - start) * 1000
        digest = hashlib.sha256(email.text.encode()).hexdigest()[:16]
        log.info("email=%s p=%.3f suspicious=%d ms=%.1f", digest, p, flagged, ms)
        return Verdict(
            label="phishing" if p >= threshold else "safe",
            p_phishing=round(p, 4),
            suspicious_chars=flagged,
            model_version=model_version,
            latency_ms=round(ms, 1),
        )

    return app
