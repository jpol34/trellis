"""Provider-dispatching LLM call helper for synthetic generation, plus the shared HTTP retry
wrapper (`post_with_retry`, vendored verbatim from laya-bench's `_post_with_retry` in
`laya_bench.eval.labeling`) that both providers' requests go through.

`call_llm` is the single seam `personas.py` and `transcript_gen.py` call instead of hand-rolling
a provider-specific request: it branches on `settings.generation_provider` and returns the
generated text for either Anthropic's Messages API or OpenAI's Chat Completions API.

A real generation run (many transcripts, several concurrent requests) routinely hits the
provider's rate limit. Without a retry, every one of those becomes a hard failure
indistinguishable from a real error. Retried only for transient statuses (429, 5xx) — a 4xx
auth/validation error is a real config problem and must still surface immediately.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from trellis.settings import settings

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BASE_DELAY_S = 2.0

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def extract_anthropic_text(response_body: dict[str, Any]) -> str:
    """Returns the first `text`-type block from an Anthropic Messages API response.

    The `content` array isn't guaranteed to start with the text block — models that use
    extended thinking prepend a `thinking`-type block first, so indexing `content[0]` directly
    breaks whenever thinking is present (on by default for some models/requests)."""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    raise ValueError(f"no text block found in Anthropic response content: {response_body!r}")


def extract_openai_text(response_body: dict[str, Any]) -> str:
    """Returns the first choice's message content from an OpenAI Chat Completions response."""
    try:
        return response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"no message content found in OpenAI response: {response_body!r}") from e


async def call_llm(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
    json_mode: bool = False,
) -> str:
    """Sends `prompt` to the configured generation provider and returns the response text.

    `json_mode=True` requests a strict-JSON response — only meaningful on the OpenAI branch
    (`response_format: {"type": "json_object"}`); the Anthropic branch has no equivalent
    request-level constraint and relies on prompt instructions alone in both modes."""
    if settings.generation_provider == "openai":
        body: dict[str, Any] = {
            "model": settings.generation_openai_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await post_with_retry(
            client,
            _OPENAI_URL,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "content-type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
        return extract_openai_text(resp.json())

    resp = await post_with_retry(
        client,
        _ANTHROPIC_URL,
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": settings.generation_claude_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    return extract_anthropic_text(resp.json())


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
