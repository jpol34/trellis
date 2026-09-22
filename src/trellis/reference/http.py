"""Retry-with-backoff HTTP POST helper for reference-arm API calls.

Vendored from laya-bench's `eval/labeling.py::_post_with_retry` rather than imported
cross-repo — trellis and laya-bench are separate deployable projects."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

# A real reference-arm run (many transcripts, several concurrent requests) routinely hits the
# provider's rate limit. Retried only for transient statuses (429, 5xx) — a 4xx auth/validation
# error is a real config problem and must still surface immediately.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BASE_DELAY_S = 2.0


async def post_with_retry(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    for attempt in range(_MAX_RETRIES + 1):
        resp = await client.post(url, **kwargs)
        if resp.status_code not in _RETRYABLE_STATUSES or attempt == _MAX_RETRIES:
            resp.raise_for_status()
            return resp
        # Retry-After may be either delay-seconds or an HTTP-date (RFC 9110) — only the
        # numeric form is actionable here, so fall back to exponential backoff for a date string
        # rather than letting float() raise and turn a retryable status into a hard failure.
        retry_after = resp.headers.get("retry-after")
        try:
            delay = float(retry_after) if retry_after else _BASE_DELAY_S * (2**attempt)
        except ValueError:
            delay = _BASE_DELAY_S * (2**attempt)
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # loop always returns or raises
