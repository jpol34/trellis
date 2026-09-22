from __future__ import annotations

import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient

import trellis.reference.jev_arm as jev_arm_module
from trellis.reference.base import ARMS
from trellis.reference.jev_arm import QUESTION_NAME, JevArm
from trellis.reference.wire import candidates_to_criteria
from trellis.schema.types import FieldSpec
from trellis.settings import settings

FIELD = FieldSpec(
    name="issue_type",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="issue_types",
)


def _mock_response(choice_key: str, confidence: float, probabilities: dict[str, float]):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1",
                "answers": {
                    QUESTION_NAME: {
                        "type": "choice",
                        "choice": choice_key,
                        "confidence": confidence,
                        "probabilities": probabilities,
                    }
                },
                "usage": {"input_tokens": 42, "output_tokens": 3},
            },
        )

    return handler, httpx2.MockTransport(handler)


@pytest.fixture(autouse=True)
def _typesafe_settings(monkeypatch):
    monkeypatch.setattr(settings, "typesafe_api_key", "test-key")
    monkeypatch.setattr(settings, "jev_base_url", "https://jev.example.test")


@pytest.fixture
def _patched_transport(monkeypatch):
    """`JevArm.answer` constructs its own `AsyncTypeSafeClient()` internally — patch that
    construction to inject a mock transport for the duration of the test."""

    def _install(transport: httpx2.MockTransport):
        class _WrappedClient(AsyncTypeSafeClient):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(jev_arm_module, "AsyncTypeSafeClient", _WrappedClient)

    return _install


def test_jev_registered_in_arms():
    assert isinstance(ARMS["jev"], JevArm)
    assert ARMS["jev"].name == "jev"


async def test_answer_maps_chosen_key_back_to_index(_patched_transport):
    candidates = ["billing", "technical", "not mentioned"]
    handler, transport = _mock_response("candidate_2", 0.83, {"candidate_2": 0.83})
    _patched_transport(transport)

    result = await ARMS["jev"].answer("some transcript", FIELD, candidates)

    assert result["chosen_index"] == 2
    assert result["value"] == "not mentioned"
    assert result["confidence"] == 0.83
    assert result["input_tokens"] == 42
    assert result["output_tokens"] == 3


async def test_criteria_sent_matches_candidate_order(_patched_transport):
    candidates = ["technical", "billing", "not mentioned"]
    captured_bodies = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1",
                "answers": {
                    QUESTION_NAME: {
                        "type": "choice",
                        "choice": "candidate_0",
                        "confidence": 0.5,
                        "probabilities": {"candidate_0": 0.5},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    _patched_transport(httpx2.MockTransport(handler))

    original_candidates = list(candidates)
    await ARMS["jev"].answer("transcript", FIELD, candidates)

    assert candidates == original_candidates  # arm never reorders/reshuffles its input

    sent_criteria = captured_bodies[0]["questions"][QUESTION_NAME]["criteria"]
    assert sent_criteria == candidates_to_criteria(original_candidates)


async def test_answer_raises_when_candidates_is_none():
    with pytest.raises(ValueError):
        await ARMS["jev"].answer("transcript", FIELD, None)


async def test_answer_raises_clear_error_when_answer_missing(_patched_transport):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1",
                "answers": {},
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        )

    _patched_transport(httpx2.MockTransport(handler))

    with pytest.raises(ValueError, match=QUESTION_NAME):
        await ARMS["jev"].answer("transcript", FIELD, ["billing", "technical"])


async def test_answer_raises_clear_error_when_chosen_index_out_of_range(_patched_transport):
    handler, transport = _mock_response("candidate_7", 0.9, {"candidate_7": 0.9})
    _patched_transport(transport)

    with pytest.raises(ValueError, match="out of range"):
        await ARMS["jev"].answer("transcript", FIELD, ["billing", "technical"])
