"""Shared HTTP retry wrapper for LLM API calls, vendored verbatim from laya-bench's
`_post_with_retry` (`laya_bench.eval.labeling`) so this project's generation calls get the same
battle-tested retry/backoff behavior instead of a parallel reimplementation drifting out of sync.

A real generation run (many transcripts, several concurrent requests) routinely hits Anthropic's
rate limit. Without a retry, every one of those becomes a hard failure indistinguishable from a
real error. Retried only for transient statuses (429, 5xx) — a 4xx auth/validation error is a real
config problem and must still surface immediately.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BASE_DELAY_S = 2.0


def extract_anthropic_text(response_body: dict[str, Any]) -> str:
    """Returns the first `text`-type block from an Anthropic Messages API response.

    The `content` array isn't guaranteed to start with the text block — models that use
    extended thinking prepend a `thinking`-type block first, so indexing `content[0]` directly
    breaks whenever thinking is present (on by default for some models/requests)."""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    raise ValueError(f"no text block found in Anthropic response content: {response_body!r}")


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
