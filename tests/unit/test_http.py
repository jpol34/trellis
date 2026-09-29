from __future__ import annotations

import json

import httpx
import pytest

import trellis.generation.http as http_module
from trellis.generation.http import (
    call_llm,
    extract_anthropic_text,
    extract_openai_text,
    post_with_retry,
)
from trellis.settings import settings


def test_extract_anthropic_text_skips_leading_thinking_block():
    body = {
        "content": [
            {"type": "thinking", "thinking": "", "signature": "abc"},
            {"type": "text", "text": "the actual answer"},
        ]
    }
    assert extract_anthropic_text(body) == "the actual answer"


def test_extract_anthropic_text_works_when_text_is_first_block():
    body = {"content": [{"type": "text", "text": "the actual answer"}]}
    assert extract_anthropic_text(body) == "the actual answer"


def test_extract_anthropic_text_raises_when_no_text_block():
    body = {"content": [{"type": "thinking", "thinking": "", "signature": "abc"}]}
    with pytest.raises(ValueError):
        extract_anthropic_text(body)


async def _no_sleep(_delay: float) -> None:
    return None


async def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await post_with_retry(client, "https://example.test/x")

    assert resp.status_code == 200
    assert calls["n"] == 3


async def test_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await post_with_retry(client, "https://example.test/x")

    assert resp.status_code == 200
    assert calls["n"] == 2


async def test_raises_immediately_on_4xx(monkeypatch):
    monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await post_with_retry(client, "https://example.test/x")

    assert calls["n"] == 1


async def test_honors_numeric_retry_after_header(monkeypatch):
    monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
    sleep_calls = []

    async def _recording_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(http_module.asyncio, "sleep", _recording_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await post_with_retry(client, "https://example.test/x")

    assert resp.status_code == 200
    assert sleep_calls == [7.0]


async def test_falls_back_to_backoff_on_http_date_retry_after(monkeypatch):
    monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(
                429,
                headers={"retry-after": "Mon, 21 Sep 2026 12:00:00 GMT"},
                json={"error": "rate limited"},
            )
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await post_with_retry(client, "https://example.test/x")

    assert resp.status_code == 200
    assert calls["n"] == 2


async def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(http_module, "_MAX_RETRIES", 2)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": "rate limited"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await post_with_retry(client, "https://example.test/x")

    assert calls["n"] == 3  # initial attempt + 2 retries


def test_extract_openai_text_returns_message_content():
    body = {"choices": [{"message": {"content": "the actual answer"}}]}
    assert extract_openai_text(body) == "the actual answer"


def test_extract_openai_text_raises_on_empty_choices():
    with pytest.raises(ValueError):
        extract_openai_text({"choices": []})


def test_extract_openai_text_raises_on_missing_choices():
    with pytest.raises(ValueError):
        extract_openai_text({})


def test_extract_openai_text_raises_on_malformed_message():
    with pytest.raises(ValueError):
        extract_openai_text({"choices": [{"message": {}}]})


async def test_call_llm_anthropic_dispatch_matches_todays_exact_request_shape(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "anthropic")
    monkeypatch.setattr(settings, "generation_claude_model", "claude-test-model")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key-123")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi there"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # json_mode=True must have no effect on the Anthropic branch's request shape.
        result = await call_llm(
            client, "a prompt", max_tokens=777, timeout=30.0, json_mode=True
        )

    assert result == "hi there"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "anthropic-key-123"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"] == {
        "model": "claude-test-model",
        "max_tokens": 777,
        "messages": [{"role": "user", "content": "a prompt"}],
    }
    assert "response_format" not in seen["body"]


async def test_call_llm_openai_dispatch_sets_response_format_only_when_json_mode(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "openai")
    monkeypatch.setattr(settings, "generation_openai_model", "gpt-test-model")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key-456")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "generated text"}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await call_llm(client, "a prompt", max_tokens=500, timeout=30.0, json_mode=True)

    assert result == "generated text"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer openai-key-456"
    assert seen["body"]["model"] == "gpt-test-model"
    assert seen["body"]["messages"] == [{"role": "user", "content": "a prompt"}]
    assert seen["body"]["response_format"] == {"type": "json_object"}


async def test_call_llm_openai_dispatch_omits_response_format_when_not_json_mode(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "openai")
    monkeypatch.setattr(settings, "generation_openai_model", "gpt-test-model")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key-456")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "prose transcript"}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await call_llm(client, "a prompt", max_tokens=500, timeout=30.0)

    assert result == "prose transcript"
    assert "response_format" not in seen["body"]
