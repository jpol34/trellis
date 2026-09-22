from __future__ import annotations

import pytest

import trellis.serving.handler as handler_module
from trellis.model.trellis_model import DiscriminationResult
from trellis.schema.types import FieldSpec


class _FakeModel:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.batch_calls: list[tuple[str, list[tuple[FieldSpec, list[str]]]]] = []
        self.next_results: list[DiscriminationResult | Exception] = []

    def discriminate_batch(
        self, transcript: str, items: list[tuple[FieldSpec, list[str]]]
    ) -> list[DiscriminationResult | Exception]:
        self.batch_calls.append((transcript, items))
        return self.next_results


@pytest.fixture
def fake_model(monkeypatch) -> _FakeModel:
    """`handler._get_model` lazily constructs `TrellisModel` on first use (see
    `trellis_arm.py`'s `_get_backend` for the pattern this mirrors) — bypass that construction
    entirely by seeding the module-level `_model` singleton directly, so tests never need a real
    checkpoint or GPU."""
    instance = _FakeModel()
    monkeypatch.setattr(handler_module, "_model", instance)
    return instance


def _event(items: list[dict]) -> dict:
    return {"input": {"transcript": "some transcript", "items": items}}


def test_normal_multi_item_request_translates_in_order(fake_model):
    fake_model.next_results = [
        DiscriminationResult(chosen_index=0, chosen_value="leak", confidence=0.9),
        DiscriminationResult(chosen_index=1, chosen_value="high", confidence=0.5),
    ]
    event = _event(
        [
            {"field_name": "issue_type", "candidates": ["leak", "noise complaint"]},
            {"field_name": "urgency", "candidates": ["low", "high"]},
        ]
    )

    result = handler_module.handler(event)

    assert result["device_used"] == "cuda"
    assert result["results"] == [
        {"chosen_index": 0, "chosen_value": "leak", "confidence": 0.9},
        {"chosen_index": 1, "chosen_value": "high", "confidence": 0.5},
    ]

    transcript, items = fake_model.batch_calls[0]
    assert transcript == "some transcript"
    assert [field.name for field, _candidates in items] == ["issue_type", "urgency"]
    assert [candidates for _field, candidates in items] == [
        ["leak", "noise complaint"],
        ["low", "high"],
    ]


def test_per_item_error_isolated_from_siblings(fake_model):
    fake_model.next_results = [
        ValueError("bad candidate index"),
        DiscriminationResult(chosen_index=0, chosen_value="low", confidence=0.42),
    ]
    event = _event(
        [
            {"field_name": "issue_type", "candidates": ["leak", "noise complaint"]},
            {"field_name": "urgency", "candidates": ["low", "high"]},
        ]
    )

    result = handler_module.handler(event)

    assert result["results"] == [
        {"error": "bad candidate index"},
        {"chosen_index": 0, "chosen_value": "low", "confidence": 0.42},
    ]


def test_device_used_reflects_resolved_model_device(fake_model):
    fake_model.device = "cpu"  # simulates a GPU->CPU fallback resolved during the call
    fake_model.next_results = [
        DiscriminationResult(chosen_index=0, chosen_value="leak", confidence=0.9)
    ]
    event = _event([{"field_name": "issue_type", "candidates": ["leak", "noise complaint"]}])

    result = handler_module.handler(event)

    assert result["device_used"] == "cpu"


def test_empty_items_does_not_crash(fake_model):
    fake_model.next_results = []
    event = _event([])

    result = handler_module.handler(event)

    assert result == {"device_used": "cuda", "results": []}


@pytest.mark.parametrize(
    "event",
    [
        {},  # missing "input" entirely
        {"input": {"items": []}},  # missing "transcript"
        {"input": {"transcript": "t", "items": [{"candidates": ["a"]}]}},  # item missing field_name
        {"input": {"transcript": "t", "items": [{"field_name": "x"}]}},  # item missing candidates
    ],
)
def test_malformed_input_raises_instead_of_misbehaving(fake_model, event):
    # A malformed event isn't caught here — RunPod's own job runner (confirmed by review against
    # the installed `runpod` package) wraps the whole `handler()` call and turns any unhandled
    # exception into a clean `status: "FAILED"` response, same as a model-construction failure.
    # This test exists to pin that "fails loudly" behavior, not to add handling for it.
    with pytest.raises(KeyError):
        handler_module.handler(event)


def test_get_model_constructs_once_and_caches(monkeypatch):
    instances: list[_FakeModel] = []

    def _fake_ctor(checkpoint_dir: str, device: str) -> _FakeModel:
        instance = _FakeModel(device=device)
        instances.append(instance)
        return instance

    monkeypatch.setattr(handler_module, "_model", None)
    monkeypatch.setattr(handler_module, "TrellisModel", _fake_ctor)

    first = handler_module._get_model()
    second = handler_module._get_model()

    assert first is second
    assert len(instances) == 1
    assert instances[0].device == "cuda"
