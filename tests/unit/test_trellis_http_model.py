"""Unit tests for `TrellisHttpModel`'s submit-and-poll RunPod client logic, via a mocked HTTP
transport — no real RunPod endpoint needed (see `tests/unit/test_gpt51_arm.py` for the same
mocking pattern used against a different reference arm's HTTP calls)."""

from __future__ import annotations

import json

import httpx
import pytest

import trellis.model.trellis_http_model as trellis_http_model_module
from trellis.model.trellis_http_model import (
    RunPodJobFailedError,
    RunPodPollTimeoutError,
    TrellisHttpModel,
)
from trellis.model.trellis_model import DiscriminationResult
from trellis.schema.types import FieldSpec

FIELD_A = FieldSpec(
    name="issue_type",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="issue_types",
)
FIELD_B = FieldSpec(
    name="urgency",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="urgency_levels",
)
CANDIDATES_A = ["leak", "noise complaint", "not mentioned"]
CANDIDATES_B = ["high", "low", "not mentioned"]

ENDPOINT_ID = "ep-abc123"


def _mock_client(monkeypatch, handler) -> list[httpx.Request]:
    """Patches `httpx.AsyncClient()` construction inside the module under test to route
    through a MockTransport, mirroring `test_gpt51_arm.py::_mock_client`."""
    requests: list[httpx.Request] = []

    def _record_and_handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    real_async_client = httpx.AsyncClient

    def _fake_async_client(*args, **kwargs) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(_record_and_handle))

    monkeypatch.setattr(trellis_http_model_module.httpx, "AsyncClient", _fake_async_client)
    return requests


def _route(monkeypatch, *, run_response: dict, status_responses: list[dict]):
    """Routes `/run` to `run_response` and successive `/status/<id>` calls to each entry of
    `status_responses` in order (the last entry repeats if polled more times than provided)."""
    status_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json=run_response)
        assert "/status/" in request.url.path
        i = min(status_calls["n"], len(status_responses) - 1)
        status_calls["n"] += 1
        return httpx.Response(200, json=status_responses[i])

    return _mock_client(monkeypatch, handler), status_calls


def test_submit_request_shape(monkeypatch):
    requests, _ = _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[
            {
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "device_used": "cuda",
                    "results": [
                        {"chosen_index": 0, "chosen_value": "leak", "confidence": 0.9}
                    ],
                },
            }
        ],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    model.discriminate_batch("some transcript", [(FIELD_A, CANDIDATES_A)])

    submit = next(r for r in requests if r.url.path.endswith("/run"))
    assert submit.url.path == f"/v2/{ENDPOINT_ID}/run"
    body = json.loads(submit.content)
    assert body == {
        "input": {
            "transcript": "some transcript",
            "items": [{"field_name": "issue_type", "candidates": CANDIDATES_A}],
        }
    }


def test_completed_response_parsed_into_discrimination_results(monkeypatch):
    _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[
            {
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "device_used": "cuda",
                    "results": [
                        {"chosen_index": 0, "chosen_value": "leak", "confidence": 0.9},
                        {"chosen_index": 1, "chosen_value": "low", "confidence": 0.7},
                    ],
                },
            }
        ],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    results = model.discriminate_batch(
        "t", [(FIELD_A, CANDIDATES_A), (FIELD_B, CANDIDATES_B)]
    )

    assert results == [
        DiscriminationResult(chosen_index=0, chosen_value="leak", confidence=0.9),
        DiscriminationResult(chosen_index=1, chosen_value="low", confidence=0.7),
    ]
    assert model.last_device_used == "cuda"


def test_polls_through_in_progress_before_completed(monkeypatch):
    monkeypatch.setattr(trellis_http_model_module.settings, "trellis_http_poll_interval_seconds", 0)
    _, status_calls = _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[
            {"id": "job-1", "status": "IN_QUEUE"},
            {"id": "job-1", "status": "IN_PROGRESS"},
            {
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "device_used": "cpu",
                    "results": [
                        {"chosen_index": 0, "chosen_value": "leak", "confidence": 0.5}
                    ],
                },
            },
        ],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    results = model.discriminate_batch("t", [(FIELD_A, CANDIDATES_A)])

    assert status_calls["n"] == 3
    assert results[0].chosen_value == "leak"
    assert model.last_device_used == "cpu"


def test_failed_status_raises_runpod_job_failed_error(monkeypatch):
    _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[
            {"id": "job-1", "status": "FAILED", "error": "worker OOM during model construction"}
        ],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    with pytest.raises(RunPodJobFailedError, match="worker OOM"):
        model.discriminate_batch("t", [(FIELD_A, CANDIDATES_A)])


def test_per_item_error_isolated_to_its_position(monkeypatch):
    _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[
            {
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "device_used": "cuda",
                    "results": [
                        {"error": "malformed answer from model"},
                        {"chosen_index": 0, "chosen_value": "high", "confidence": 0.8},
                    ],
                },
            }
        ],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    results = model.discriminate_batch(
        "t", [(FIELD_A, CANDIDATES_A), (FIELD_B, CANDIDATES_B)]
    )

    assert isinstance(results[0], ValueError)
    assert isinstance(results[1], DiscriminationResult)
    assert results[1].chosen_value == "high"


def test_out_of_range_chosen_index_becomes_value_error(monkeypatch):
    _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[
            {
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "device_used": "cuda",
                    "results": [
                        {"chosen_index": 99, "chosen_value": "?", "confidence": 0.5}
                    ],
                },
            }
        ],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    results = model.discriminate_batch("t", [(FIELD_A, CANDIDATES_A)])

    assert isinstance(results[0], ValueError)


def test_poll_budget_exhaustion_raises_timeout_error(monkeypatch):
    monkeypatch.setattr(trellis_http_model_module.settings, "trellis_http_poll_interval_seconds", 0)
    monkeypatch.setattr(trellis_http_model_module.settings, "trellis_http_poll_timeout_seconds", 0)
    _route(
        monkeypatch,
        run_response={"id": "job-1", "status": "IN_QUEUE"},
        status_responses=[{"id": "job-1", "status": "IN_PROGRESS"}],
    )
    model = TrellisHttpModel(ENDPOINT_ID)

    with pytest.raises(RunPodPollTimeoutError):
        model.discriminate_batch("t", [(FIELD_A, CANDIDATES_A)])


def test_submit_transport_failure_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad api key"})

    _mock_client(monkeypatch, handler)
    model = TrellisHttpModel(ENDPOINT_ID)

    with pytest.raises(httpx.HTTPStatusError):
        model.discriminate_batch("t", [(FIELD_A, CANDIDATES_A)])


def test_submit_retries_on_429_then_succeeds(monkeypatch):
    async def _fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("trellis.reference.http.asyncio.sleep", _fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/run"):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json={"id": "job-1", "status": "IN_QUEUE"})
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "COMPLETED",
                "output": {
                    "device_used": "cuda",
                    "results": [
                        {"chosen_index": 0, "chosen_value": "leak", "confidence": 0.9}
                    ],
                },
            },
        )

    _mock_client(monkeypatch, handler)
    model = TrellisHttpModel(ENDPOINT_ID)

    results = model.discriminate_batch("t", [(FIELD_A, CANDIDATES_A)])

    assert calls["n"] == 2
    assert results[0].chosen_value == "leak"
