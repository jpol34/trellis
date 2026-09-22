from __future__ import annotations

import asyncio

import pytest

import trellis.reference.trellis_arm as trellis_arm_module
from trellis.model.trellis_model import DiscriminationResult
from trellis.reference.base import ARMS
from trellis.reference.trellis_arm import TrellisArm
from trellis.schema.types import FieldSpec

FIELD = FieldSpec(
    name="issue_type",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="issue_types",
)

CANDIDATES = ["leak", "noise complaint", "not mentioned"]


class _FakeModel:
    def __init__(self, checkpoint_dir: str, device: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.batch_calls: list[tuple[str, list[tuple[FieldSpec, list[str]]]]] = []
        self.raise_on_batch: Exception | None = None

    def discriminate_batch(
        self, transcript: str, items: list[tuple[FieldSpec, list[str]]]
    ) -> list[DiscriminationResult | Exception]:
        self.batch_calls.append((transcript, items))
        if self.raise_on_batch is not None:
            raise self.raise_on_batch
        return [
            DiscriminationResult(chosen_index=0, chosen_value=candidates[0], confidence=0.77)
            for _field, candidates in items
        ]


@pytest.fixture
def _patched_model(monkeypatch):
    """`TrellisArm._get_backend` constructs its own `TrellisModel` lazily on first use (when
    `trellis_backend="local"`, the default) — patch that construction so tests never need a
    real checkpoint on disk."""
    instances: list[_FakeModel] = []

    def _fake_ctor(checkpoint_dir: str, device: str) -> _FakeModel:
        instance = _FakeModel(checkpoint_dir, device)
        instances.append(instance)
        return instance

    monkeypatch.setattr(trellis_arm_module, "TrellisModel", _fake_ctor)
    return instances


def test_trellis_registered_in_arms():
    assert isinstance(ARMS["trellis"], TrellisArm)
    assert ARMS["trellis"].name == "trellis"
    assert ARMS["trellis"].mode == "closed_set"


async def test_answer_delegates_to_trellis_model(_patched_model):
    arm = TrellisArm()

    result = await arm.answer("some transcript", FIELD, CANDIDATES)

    assert result["value"] == "leak"
    assert result["chosen_index"] == 0
    assert result["confidence"] == 0.77
    assert len(_patched_model) == 1
    assert _patched_model[0].batch_calls == [
        ("some transcript", [(FIELD, CANDIDATES)])
    ]


async def test_model_is_constructed_lazily_and_reused(_patched_model):
    arm = TrellisArm()
    assert _patched_model == []  # not constructed at arm construction time

    await arm.answer("t1", FIELD, CANDIDATES)
    await arm.answer("t2", FIELD, CANDIDATES)

    assert len(_patched_model) == 1  # same model instance reused across calls


async def test_concurrent_first_calls_construct_model_once(_patched_model):
    arm = TrellisArm()

    await asyncio.gather(
        arm.answer("t1", FIELD, CANDIDATES),
        arm.answer("t2", FIELD, CANDIDATES),
        arm.answer("t3", FIELD, CANDIDATES),
    )

    assert len(_patched_model) == 1  # the construct-lock prevented a double build


async def test_answer_raises_when_candidates_is_none(_patched_model):
    arm = TrellisArm()

    with pytest.raises(ValueError):
        await arm.answer("transcript", FIELD, None)


OTHER_FIELD = FieldSpec(
    name="urgency",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="urgency_levels",
)
OTHER_CANDIDATES = ["high", "low", "not mentioned"]


async def test_concurrent_answers_for_one_transcript_coalesce_into_one_batch_call(
    _patched_model,
):
    arm = TrellisArm()

    results = await asyncio.gather(
        arm.answer("shared transcript", FIELD, CANDIDATES),
        arm.answer("shared transcript", OTHER_FIELD, OTHER_CANDIDATES),
    )

    assert len(_patched_model) == 1
    assert len(_patched_model[0].batch_calls) == 1
    transcript, items = _patched_model[0].batch_calls[0]
    assert transcript == "shared transcript"
    assert items == [(FIELD, CANDIDATES), (OTHER_FIELD, OTHER_CANDIDATES)]
    assert [r["value"] for r in results] == ["leak", "high"]


async def test_different_transcripts_never_coalesce(_patched_model):
    arm = TrellisArm()

    await asyncio.gather(
        arm.answer("transcript one", FIELD, CANDIDATES),
        arm.answer("transcript two", OTHER_FIELD, OTHER_CANDIDATES),
    )

    assert len(_patched_model) == 1
    assert len(_patched_model[0].batch_calls) == 2
    transcripts = {transcript for transcript, _items in _patched_model[0].batch_calls}
    assert transcripts == {"transcript one", "transcript two"}


async def test_late_arrival_after_flush_starts_a_fresh_batch(_patched_model):
    arm = TrellisArm()

    await arm.answer("shared transcript", FIELD, CANDIDATES)
    await arm.answer("shared transcript", OTHER_FIELD, OTHER_CANDIDATES)

    assert len(_patched_model) == 1
    assert len(_patched_model[0].batch_calls) == 2
    for _transcript, items in _patched_model[0].batch_calls:
        assert len(items) == 1


async def test_batch_level_exception_resolves_every_waiting_future(_patched_model):
    arm = TrellisArm()
    await arm._get_backend()
    _patched_model[0].raise_on_batch = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.gather(
            arm.answer("shared transcript", FIELD, CANDIDATES),
            arm.answer("shared transcript", OTHER_FIELD, OTHER_CANDIDATES),
        )


async def test_model_construction_failure_during_flush_resolves_every_follower(monkeypatch):
    """A checkpoint-load failure inside `_flush`'s leader must not leave concurrent followers
    hanging on `await future` forever — regression test for the construction call having lived
    outside `_flush`'s try/except."""

    def _failing_ctor(checkpoint_dir: str, device: str):
        raise RuntimeError("checkpoint load failed")

    monkeypatch.setattr(trellis_arm_module, "TrellisModel", _failing_ctor)
    arm = TrellisArm()

    async def _await_with_timeout(coro):
        return await asyncio.wait_for(coro, timeout=5)

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        await asyncio.gather(
            _await_with_timeout(arm.answer("shared transcript", FIELD, CANDIDATES)),
            _await_with_timeout(
                arm.answer("shared transcript", OTHER_FIELD, OTHER_CANDIDATES)
            ),
        )


async def test_backend_defaults_to_local_trellis_model(_patched_model):
    arm = TrellisArm()

    backend = await arm._get_backend()

    assert len(_patched_model) == 1
    assert backend is _patched_model[0]


async def test_http_backend_selected_when_configured(monkeypatch):
    monkeypatch.setattr(trellis_arm_module.settings, "trellis_backend", "http")
    monkeypatch.setattr(trellis_arm_module.settings, "trellis_http_endpoint_id", "ep-123")

    instances: list[trellis_arm_module.TrellisHttpModel] = []
    real_ctor = trellis_arm_module.TrellisHttpModel

    def _spy_ctor(endpoint_id: str):
        instance = real_ctor(endpoint_id)
        instances.append(instance)
        return instance

    monkeypatch.setattr(trellis_arm_module, "TrellisHttpModel", _spy_ctor)

    arm = TrellisArm()
    backend = await arm._get_backend()

    assert len(instances) == 1
    assert backend is instances[0]
    assert instances[0]._endpoint_id == "ep-123"
