from __future__ import annotations

import json

import httpx
import pytest

from trellis.reference import gpt51_arm
from trellis.reference.base import ARMS
from trellis.reference.gpt51_arm import GPT51Arm, _parse_response
from trellis.schema.types import FieldSpec

FIELD = FieldSpec(
    name="callback_number",
    match_type="normalized_phone",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool=None,
)

TRANSCRIPT = (
    "Agent: Can I get a number to reach you at?\n"
    "Caller: Sure, it's 555-123-4567.\n"
    "Agent: Great, thanks."
)


def _mock_client(monkeypatch, handler) -> list[httpx.Request]:
    """Patches gpt51_arm's `httpx.AsyncClient()` construction to route through a MockTransport,
    since `GPT51Arm.answer` builds its own client internally rather than taking one as a
    parameter."""
    requests: list[httpx.Request] = []

    def _record_and_handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    real_async_client = httpx.AsyncClient

    def _fake_async_client(*args, **kwargs) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(_record_and_handle))

    monkeypatch.setattr(gpt51_arm.httpx, "AsyncClient", _fake_async_client)
    return requests


def _chat_completion(content: str, usage: dict | None = None) -> dict:
    body: dict = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def test_arm_is_registered_into_arms_at_import_time() -> None:
    assert "gpt-5.1" in ARMS
    assert isinstance(ARMS["gpt-5.1"], GPT51Arm)


async def test_answer_parses_evidence_first_found_response(monkeypatch) -> None:
    payload = json.dumps(
        {
            "evidence": "it's 555-123-4567",
            "status": "found",
            "value": "555-123-4567",
            "confidence": 0.95,
        }
    )
    usage = {"prompt_tokens": 120, "completion_tokens": 30}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion(payload, usage=usage))

    requests = _mock_client(monkeypatch, handler)

    arm = GPT51Arm()
    result = await arm.answer(TRANSCRIPT, FIELD, None)

    assert result == {
        "value": "555-123-4567",
        "chosen_index": None,
        "confidence": 0.95,
        "input_tokens": 120,
        "output_tokens": 30,
    }
    assert len(requests) == 1
    sent = json.loads(requests[0].content)
    assert sent["model"] == "gpt-5.1"
    assert "callback_number" in sent["messages"][0]["content"]
    assert TRANSCRIPT in sent["messages"][0]["content"]


async def test_answer_absent_field_produces_abstain_output(monkeypatch) -> None:
    payload = json.dumps({"evidence": "", "status": "insufficient"})
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion(payload))

    _mock_client(monkeypatch, handler)

    arm = GPT51Arm()
    no_field_transcript = "Agent: How can I help? Caller: I have a billing question."
    result = await arm.answer(no_field_transcript, FIELD, None)

    assert result == {
        "value": None,
        "chosen_index": None,
        "confidence": 0.0,
        "input_tokens": None,
        "output_tokens": None,
    }


async def test_answer_tolerates_surrounding_prose_around_json(monkeypatch) -> None:
    payload = (
        "Sure, here is my answer:\n"
        '```json\n{"evidence": "555-123-4567", "status": "found", "value": "555-123-4567", '
        '"confidence": 0.8}\n```'
    )
    _mock_client(monkeypatch, lambda req: httpx.Response(200, json=_chat_completion(payload)))

    arm = GPT51Arm()
    result = await arm.answer(TRANSCRIPT, FIELD, None)
    assert result["value"] == "555-123-4567"


async def test_answer_uses_configured_model_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(gpt51_arm.settings, "reference_gpt_model", "gpt-5.1-custom")
    payload = json.dumps(
        {"evidence": "x", "status": "found", "value": "x", "confidence": 0.5}
    )
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion(payload))

    requests = _mock_client(monkeypatch, handler)

    arm = GPT51Arm()
    await arm.answer(TRANSCRIPT, FIELD, None)

    sent = json.loads(requests[0].content)
    assert sent["model"] == "gpt-5.1-custom"


async def test_answer_retries_on_429_then_succeeds(monkeypatch) -> None:
    async def _fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("trellis.reference.http.asyncio.sleep", _fake_sleep)

    calls = {"n": 0}
    payload = json.dumps(
        {"evidence": "555-123-4567", "status": "found", "value": "555-123-4567", "confidence": 0.9}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=_chat_completion(payload))

    _mock_client(monkeypatch, handler)

    arm = GPT51Arm()
    result = await arm.answer(TRANSCRIPT, FIELD, None)

    assert calls["n"] == 2
    assert result["value"] == "555-123-4567"


async def test_answer_raises_immediately_on_4xx(monkeypatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad api key"})

    _mock_client(monkeypatch, handler)

    arm = GPT51Arm()
    with pytest.raises(httpx.HTTPStatusError):
        await arm.answer(TRANSCRIPT, FIELD, None)
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "raw",
    [
        '{"evidence": "x", "status": "maybe", "value": "x", "confidence": 0.5}',
        '{"evidence": "x", "status": "found", "value": "", "confidence": 0.5}',
        '{"evidence": "", "status": "found", "value": "x", "confidence": 0.5}',
        '{"evidence": "x", "status": "found", "value": "x", "confidence": 1.5}',
        '{"evidence": "x", "status": "found", "value": "x", "confidence": "high"}',
        "not json at all",
    ],
)
def test_parse_response_rejects_malformed_payloads(raw: str) -> None:
    with pytest.raises(ValueError):
        _parse_response(raw)


def test_parse_response_accepts_boundary_confidence_values() -> None:
    low = _parse_response('{"evidence": "e", "status": "found", "value": "v", "confidence": 0}')
    high = _parse_response('{"evidence": "e", "status": "found", "value": "v", "confidence": 1}')
    assert low["confidence"] == 0.0
    assert high["confidence"] == 1.0
