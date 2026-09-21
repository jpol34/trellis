from __future__ import annotations

import httpx
import pytest

import trellis.generation.http as http_module
from trellis.generation.http import post_with_retry


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
