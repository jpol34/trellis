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
        self.calls: list[tuple[str, FieldSpec, list[str]]] = []

    def discriminate(
        self, transcript: str, field: FieldSpec, candidates: list[str]
    ) -> DiscriminationResult:
        self.calls.append((transcript, field, candidates))
        return DiscriminationResult(chosen_index=0, chosen_value=candidates[0], confidence=0.77)


@pytest.fixture
def _patched_model(monkeypatch):
    """`TrellisArm._get_model` constructs its own `TrellisModel` lazily on first use — patch
    that construction so tests never need a real checkpoint on disk."""
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
    assert _patched_model[0].calls == [("some transcript", FIELD, CANDIDATES)]


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
