from __future__ import annotations

import httpx
import pytest

from trellis.reference.http import post_with_retry


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Retries sleep for real seconds by design (Retry-After honoring, exponential backoff) —
    replace with a no-op recorder so these tests run instantly, and record the delays so the
    backoff/Retry-After logic itself is still verifiable."""
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("trellis.reference.http.asyncio.sleep", _fake_sleep)
    return delays


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_returns_response_on_immediate_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        resp = await post_with_retry(client, "https://example.test/x")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def test_retries_on_429_honoring_retry_after(no_real_sleep) -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow down"})
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        resp = await post_with_retry(client, "https://example.test/x")
    assert resp.status_code == 200
    assert calls["n"] == 2
    assert no_real_sleep == [3.0]


async def test_retries_on_5xx_with_exponential_backoff_when_no_retry_after(no_real_sleep) -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        resp = await post_with_retry(client, "https://example.test/x")
    assert resp.status_code == 200
    assert calls["n"] == 3
    assert no_real_sleep == [2.0, 4.0]


async def test_non_numeric_retry_after_falls_back_to_exponential_backoff(no_real_sleep) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"})
            if len(no_real_sleep) == 0
            else httpx.Response(200, json={"ok": True})
        )

    async with _client(handler) as client:
        resp = await post_with_retry(client, "https://example.test/x")
    assert resp.status_code == 200
    assert no_real_sleep == [2.0]


async def test_raises_immediately_on_4xx_without_retry(no_real_sleep) -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await post_with_retry(client, "https://example.test/x")
    assert calls["n"] == 1
    assert no_real_sleep == []


async def test_raises_after_exhausting_retries_on_persistent_429(no_real_sleep) -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await post_with_retry(client, "https://example.test/x")
    assert calls["n"] == 6  # initial attempt + 5 retries
    assert len(no_real_sleep) == 5
