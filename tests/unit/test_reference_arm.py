from __future__ import annotations

import random

import pytest

from trellis.reference.base import ARMS, ArmAnswer, register_arm
from trellis.reference.wire import candidates_to_criteria, criteria_key_to_index
from trellis.schema.types import FieldSpec

FIELD = FieldSpec(
    name="issue_type",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="issue_types",
)


@pytest.mark.parametrize("size", [2, 5, 10])
def test_criteria_round_trip_preserves_index(size: int) -> None:
    candidates = [f"option_{i}" for i in range(size - 1)] + ["not mentioned"]
    random.Random(size).shuffle(candidates)

    criteria = candidates_to_criteria(candidates)
    assert len(criteria) == size

    for original_index, text in enumerate(candidates):
        key = f"candidate_{original_index}"
        assert criteria[key] == text
        assert criteria_key_to_index(key) == original_index


def test_criteria_round_trip_empty_candidate_list() -> None:
    assert candidates_to_criteria([]) == {}


def test_criteria_round_trip_single_candidate() -> None:
    criteria = candidates_to_criteria(["only option"])
    assert criteria == {"candidate_0": "only option"}
    assert criteria_key_to_index("candidate_0") == 0


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "candidate_",
        "candidate_x",
        "candidateX0",
        "0_candidate",
        "candidate_-1",
        "candidate_²",
    ],
)
def test_malformed_key_raises_value_error(bad_key: str) -> None:
    with pytest.raises(ValueError):
        criteria_key_to_index(bad_key)


def test_malformed_key_message_includes_bad_key() -> None:
    with pytest.raises(ValueError) as exc_info:
        criteria_key_to_index("not_a_candidate_key")

    assert "not_a_candidate_key" in str(exc_info.value)


class DummyArm:
    name = "dummy"

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer:
        if candidates is None:
            return {"value": "extracted from transcript", "chosen_index": None, "confidence": 0.9}
        return {"value": candidates[0], "chosen_index": 0, "confidence": 0.9}


class SecondDummyArm:
    name = "dummy_two"

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer:
        return {"value": None, "chosen_index": None, "confidence": 0.5}


@pytest.fixture
def registered_arms():
    # Snapshots and clears the real registry rather than adding onto it — concrete arm modules
    # (e.g. gpt-5.1, jev) register themselves into this same process-global `ARMS` as soon as
    # they're imported anywhere in the test session, and this fixture's tests assume they're
    # the only entries, including calling `answer()` on every entry with no HTTP mocking in place.
    previous = dict(ARMS)
    ARMS.clear()
    ARMS["dummy"] = DummyArm()
    try:
        yield ARMS
    finally:
        ARMS.clear()
        ARMS.update(previous)


async def test_registered_arm_is_discovered_and_callable(registered_arms) -> None:
    results = {}
    for arm_name, arm in registered_arms.items():
        results[arm_name] = await arm.answer("some transcript", FIELD, ["a", "b", "c"])

    assert results["dummy"]["value"] == "a"
    assert results["dummy"]["chosen_index"] == 0


async def test_adding_a_second_arm_requires_no_base_changes(registered_arms) -> None:
    registered_arms["dummy_two"] = SecondDummyArm()
    try:
        assert len(registered_arms) == 2
        assert set(registered_arms) == {"dummy", "dummy_two"}

        for arm in registered_arms.values():
            assert isinstance(await arm.answer("t", FIELD, None), dict)
    finally:
        registered_arms.pop("dummy_two", None)


def test_register_arm_adds_to_registry(registered_arms) -> None:
    register_arm(SecondDummyArm())
    try:
        assert registered_arms["dummy_two"].name == "dummy_two"
    finally:
        registered_arms.pop("dummy_two", None)


def test_register_arm_rejects_duplicate_name(registered_arms) -> None:
    with pytest.raises(ValueError):
        register_arm(DummyArm())
