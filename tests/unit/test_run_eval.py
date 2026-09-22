"""Covers #15's run_eval pipeline end-to-end against mocked arms: dispatch-by-mode (closed_set
vs open_extraction candidates), 5-state resolution for every outcome, per-item error isolation
(one arm/item failure must not crash the run), and the "all fields not mentioned" edge case."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from trellis.reference.base import ArmAnswer
from trellis.schema.types import FieldSpec
from trellis.validate.run_eval import (
    _EvalCheckpoint,
    _field_specs_by_category,
    _score_one,
    load_eval_set,
    run_all,
    run_arm,
    run_eval,
)

REQUIRED_FIELD = FieldSpec(
    name="name", match_type="fuzzy", required=True, distractor_strategy="swap_similar_name"
)
OPTIONAL_FIELD = FieldSpec(
    name="pet_info",
    match_type="canonical_list",
    required=False,
    distractor_strategy="swap_sibling_value",
    value_pool="pet_policies",
)

FIELD_SPECS = {"prospect": {"name": REQUIRED_FIELD, "pet_info": OPTIONAL_FIELD}}


def _record(item_id: str, name_field: dict, pet_field: dict) -> dict:
    return {
        "item_id": item_id,
        "category": "prospect",
        "call_reason": "new_inquiry",
        "variability_tier": "clean",
        "transcript": "some transcript",
        "fields": [name_field, pet_field],
    }


NAME_FIELD_RECORD = {
    "field": "name",
    "value": "Jane Doe",
    "candidates": ["Jane Doe", "John Smith", "Amy Lee"],
    "correct_index": 0,
    "char_offset": 0,
    "distractor_strategy": "swap_similar_name",
}

PET_MENTIONED_RECORD = {
    "field": "pet_info",
    "value": "one dog",
    "candidates": ["one dog", "no pets", "two cats", "not mentioned"],
    "correct_index": 0,
    "char_offset": 10,
    "distractor_strategy": "swap_sibling_value",
}

PET_NOT_MENTIONED_RECORD = {
    "field": "pet_info",
    "value": "not mentioned",
    "candidates": ["one dog", "no pets", "two cats", "not mentioned"],
    "correct_index": 3,
    "char_offset": 10,
    "distractor_strategy": "swap_sibling_value",
}


class ClosedArm:
    name = "closed"
    mode = "closed_set"

    def __init__(self, chosen_index: int) -> None:
        self.chosen_index = chosen_index
        self.calls: list[tuple[str, FieldSpec, list[str] | None]] = []

    async def answer(self, transcript, field, candidates) -> ArmAnswer:
        self.calls.append((transcript, field, candidates))
        return ArmAnswer(
            value=candidates[self.chosen_index], chosen_index=self.chosen_index, confidence=0.8
        )


class OpenArm:
    name = "open"
    mode = "open_extraction"

    def __init__(self, value: str | None) -> None:
        self.value = value
        self.calls: list[tuple[str, FieldSpec, list[str] | None]] = []

    async def answer(self, transcript, field, candidates) -> ArmAnswer:
        self.calls.append((transcript, field, candidates))
        return ArmAnswer(value=self.value, chosen_index=None, confidence=0.7)


def _field_item(results, arm_name: str, field_name: str):
    items = results[arm_name].datasets["prospect"].items
    return next(i for i in items if i.field == field_name)


class RaisingArm:
    name = "raising"
    mode = "closed_set"

    async def answer(self, transcript, field, candidates) -> ArmAnswer:
        raise RuntimeError("simulated auth failure")


async def test_closed_set_arm_receives_candidates() -> None:
    arm = ClosedArm(chosen_index=0)
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"closed": arm})

    assert arm.calls[0][2] == NAME_FIELD_RECORD["candidates"]
    items = results["closed"].datasets["prospect"].items
    assert {i.field for i in items} == {"name", "pet_info"}


async def test_open_extraction_arm_receives_no_candidates() -> None:
    arm = OpenArm(value="Jane Doe")
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    await run_all([record], FIELD_SPECS, arms={"open": arm})

    assert arm.calls[0][2] is None


async def test_closed_set_present_correct() -> None:
    arm = ClosedArm(chosen_index=0)  # "Jane Doe", correct_index=0
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"closed": arm})
    name_item = next(i for i in results["closed"].datasets["prospect"].items if i.field == "name")
    assert name_item.state == "present_correct"


async def test_closed_set_present_incorrect() -> None:
    arm = ClosedArm(chosen_index=1)  # "John Smith", wrong
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"closed": arm})
    name_item = next(i for i in results["closed"].datasets["prospect"].items if i.field == "name")
    assert name_item.state == "present_incorrect"


async def test_closed_set_silently_dropped() -> None:
    # pet_info is genuinely stated ("one dog") but the model abstains ("not mentioned", index 3)
    arm = ClosedArm(chosen_index=3)
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"closed": arm})
    pet_item = _field_item(results, "closed", "pet_info")
    assert pet_item.state == "silently_dropped"


async def test_closed_set_hallucinated() -> None:
    # gold is genuinely "not mentioned" but the model claims a value
    arm = ClosedArm(chosen_index=0)
    record = _record("r1", NAME_FIELD_RECORD, PET_NOT_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"closed": arm})
    pet_item = _field_item(results, "closed", "pet_info")
    assert pet_item.state == "hallucinated"


async def test_closed_set_correctly_absent() -> None:
    arm = ClosedArm(chosen_index=3)  # abstains, and gold is genuinely absent too
    record = _record("r1", NAME_FIELD_RECORD, PET_NOT_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"closed": arm})
    pet_item = _field_item(results, "closed", "pet_info")
    assert pet_item.state == "correctly_absent"


async def test_open_extraction_present_correct() -> None:
    arm = OpenArm(value="Jane Doe")
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"open": arm})
    name_item = next(i for i in results["open"].datasets["prospect"].items if i.field == "name")
    assert name_item.state == "present_correct"


async def test_open_extraction_hallucinated() -> None:
    arm = OpenArm(value="some made up pet")
    record = _record("r1", NAME_FIELD_RECORD, PET_NOT_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"open": arm})
    pet_item = next(i for i in results["open"].datasets["prospect"].items if i.field == "pet_info")
    assert pet_item.state == "hallucinated"


async def test_open_extraction_correctly_absent() -> None:
    arm = OpenArm(value=None)  # abstains
    record = _record("r1", NAME_FIELD_RECORD, PET_NOT_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"open": arm})
    pet_item = next(i for i in results["open"].datasets["prospect"].items if i.field == "pet_info")
    assert pet_item.state == "correctly_absent"


async def test_arm_failure_is_isolated_not_crashed() -> None:
    arm = RaisingArm()
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"raising": arm})

    items = results["raising"].datasets["prospect"].items
    assert len(items) == 2
    for item in items:
        assert item.state is None
        assert item.error is not None
        assert "simulated auth failure" in item.error


async def test_mixed_success_and_failure_arms_both_complete() -> None:
    good = ClosedArm(chosen_index=0)
    bad = RaisingArm()
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    results = await run_all([record], FIELD_SPECS, arms={"good": good, "bad": bad})

    assert results["good"].datasets["prospect"].items[0].error is None
    assert results["bad"].datasets["prospect"].items[0].error is not None


async def test_all_fields_not_mentioned_transcript_scored_as_correctly_absent_across_arms() -> None:
    """The edge case the ticket calls out explicitly: a transcript where every optional field is
    genuinely absent must resolve as correct-abstention (not a crash, not hallucination) for both
    a closed-set arm and an open-extraction arm that correctly abstain."""
    all_absent_field_specs = {
        "prospect": {
            "pet_info": OPTIONAL_FIELD,
            "parking_need": FieldSpec(
                name="parking_need",
                match_type="canonical_list",
                required=False,
                distractor_strategy="swap_sibling_value",
                value_pool="parking_types",
            ),
        }
    }
    record = {
        "item_id": "all-absent",
        "category": "prospect",
        "call_reason": "new_inquiry",
        "variability_tier": "clean",
        "transcript": "Caller: hi, I have a general question. Agent: sure, go ahead.",
        "fields": [
            {
                "field": "pet_info",
                "value": "not mentioned",
                "candidates": ["one dog", "no pets", "two cats", "not mentioned"],
                "correct_index": 3,
                "char_offset": 0,
                "distractor_strategy": "swap_sibling_value",
            },
            {
                "field": "parking_need",
                "value": "not mentioned",
                "candidates": ["covered", "uncovered", "garage", "not mentioned"],
                "correct_index": 3,
                "char_offset": 0,
                "distractor_strategy": "swap_sibling_value",
            },
        ],
    }

    closed_arm = ClosedArm(chosen_index=3)  # always picks "not mentioned"
    open_arm = OpenArm(value=None)  # always abstains

    results = await run_all(
        [record], all_absent_field_specs, arms={"closed": closed_arm, "open": open_arm}
    )

    for arm_name in ("closed", "open"):
        items = results[arm_name].datasets["prospect"].items
        assert len(items) == 2
        for item in items:
            assert item.error is None
            assert item.state == "correctly_absent"


def test_field_specs_by_category_indexes_by_name() -> None:
    from trellis.schema.types import CategorySpec, ScenarioSpec

    category = CategorySpec(
        category="prospect", fields=[REQUIRED_FIELD], scenarios=ScenarioSpec(call_reasons=["x"])
    )
    specs = _field_specs_by_category([category])
    assert specs == {"prospect": {"name": REQUIRED_FIELD}}


def test_load_eval_set_reads_every_jsonl_in_dir(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"item_id": "a1"}\n', encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"item_id": "b1"}\n{"item_id": "b2"}\n', encoding="utf-8")

    records = load_eval_set(tmp_path)

    assert {r["item_id"] for r in records} == {"a1", "b1", "b2"}


def test_load_eval_set_raises_when_no_jsonl_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path)


class TokenArm:
    name = "priced"
    mode = "closed_set"

    async def answer(self, transcript, field, candidates) -> ArmAnswer:
        return ArmAnswer(
            value=candidates[0],
            chosen_index=0,
            confidence=0.9,
            input_tokens=100,
            output_tokens=20,
        )


async def test_score_one_carries_token_usage_into_item_result() -> None:
    arm = TokenArm()
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)
    item = await _score_one(arm, record, NAME_FIELD_RECORD, REQUIRED_FIELD)

    assert item.input_tokens == 100
    assert item.output_tokens == 20


async def test_score_one_leaves_token_usage_none_when_arm_omits_it() -> None:
    arm = ClosedArm(chosen_index=0)
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)
    item = await _score_one(arm, record, NAME_FIELD_RECORD, REQUIRED_FIELD)

    assert item.input_tokens is None
    assert item.output_tokens is None


class SlowArm:
    """A closed-set arm that sleeps `delay` seconds per item — used to distinguish concurrent
    from sequential arm execution by wall-clock time."""

    def __init__(self, delay: float, chosen_index: int = 0) -> None:
        self.name = "slow"
        self.mode = "closed_set"
        self.delay = delay
        self.chosen_index = chosen_index
        self.calls: list[str] = []

    async def answer(self, transcript, field, candidates) -> ArmAnswer:
        self.calls.append(field.name)
        await asyncio.sleep(self.delay)
        return ArmAnswer(
            value=candidates[self.chosen_index], chosen_index=self.chosen_index, confidence=0.8
        )


async def test_arms_run_concurrently_not_sequentially() -> None:
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)
    delay = 0.2
    # Each arm scores 2 fields concurrently within itself (default concurrency=5), so one arm
    # alone takes ~delay. 4 arms run sequentially would take ~4*delay; concurrently, ~delay.
    arms = {f"slow{i}": SlowArm(delay) for i in range(4)}

    started = time.perf_counter()
    await run_all([record], FIELD_SPECS, arms=arms)
    elapsed = time.perf_counter() - started

    assert elapsed < delay * 2.5


async def test_checkpoint_write_then_resume_skips_and_avoids_recall(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    first_arm = ClosedArm(chosen_index=0)
    checkpoint = _EvalCheckpoint(checkpoint_path)
    await run_arm(first_arm, [record], FIELD_SPECS, concurrency=5, checkpoint=checkpoint)
    checkpoint.close()

    assert len(first_arm.calls) == 2
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 2

    second_arm = ClosedArm(chosen_index=0)  # same arm.name ("closed") as first_arm
    resumed_checkpoint = _EvalCheckpoint(checkpoint_path)
    result = await run_arm(
        second_arm, [record], FIELD_SPECS, concurrency=5, checkpoint=resumed_checkpoint
    )
    resumed_checkpoint.close()

    assert second_arm.calls == []
    items = result.datasets["prospect"].items
    assert {i.field for i in items} == {"name", "pet_info"}
    # resuming with nothing new to do doesn't grow the checkpoint file
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 2


async def test_checkpoint_tolerates_malformed_trailing_line(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    good_row = {
        "backend_name": "closed",
        "item_id": "r1",
        "category": "prospect",
        "field": "name",
        "predicted_value": "Jane Doe",
        "gold_value": "Jane Doe",
        "confidence": 0.8,
        "latency_ms": 1.0,
        "state": "present_correct",
        "error": None,
    }
    checkpoint_path.write_text(
        json.dumps(good_row) + "\n" + '{"backend_name": "closed", "item_i', encoding="utf-8"
    )

    checkpoint = _EvalCheckpoint(checkpoint_path)

    assert checkpoint.get("closed", "r1", "name", "Jane Doe") is not None
    assert len(checkpoint.on_disk) == 1


async def test_checkpoint_stale_gold_value_is_not_reused(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    stale_row = {
        "backend_name": "closed",
        "item_id": "r1",
        "category": "prospect",
        "field": "name",
        "predicted_value": "Old Name",
        "gold_value": "Old Name",
        "confidence": 0.8,
        "latency_ms": 1.0,
        "state": "present_correct",
        "error": None,
    }
    checkpoint_path.write_text(json.dumps(stale_row) + "\n", encoding="utf-8")

    checkpoint = _EvalCheckpoint(checkpoint_path)

    # A regenerated eval set reused item_id "r1"/field "name" but with a different gold value —
    # the stale cached prediction must not be served for the new gold value.
    assert checkpoint.get("closed", "r1", "name", "Old Name") is not None
    assert checkpoint.get("closed", "r1", "name", "New Name") is None


async def test_checkpoint_does_not_cache_errored_items_and_retries_on_resume(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    failing_arm = RaisingArm()
    checkpoint = _EvalCheckpoint(checkpoint_path)
    await run_arm(failing_arm, [record], FIELD_SPECS, concurrency=5, checkpoint=checkpoint)
    checkpoint.close()

    # Nothing successful happened, so nothing should have been persisted.
    assert not checkpoint_path.exists() or checkpoint_path.read_text(encoding="utf-8") == ""

    retry_arm = ClosedArm(chosen_index=0)  # a different (now-working) arm named "raising"
    retry_arm.name = "raising"
    resumed_checkpoint = _EvalCheckpoint(checkpoint_path)
    result = await run_arm(
        retry_arm, [record], FIELD_SPECS, concurrency=5, checkpoint=resumed_checkpoint
    )
    resumed_checkpoint.close()

    # The retry actually ran the arm again (not served a cached error) and succeeded.
    assert len(retry_arm.calls) == 2
    items = result.datasets["prospect"].items
    assert all(item.error is None for item in items)


async def test_no_checkpoint_means_no_resume_skip() -> None:
    arm = ClosedArm(chosen_index=0)
    record = _record("r1", NAME_FIELD_RECORD, PET_MENTIONED_RECORD)

    await run_all([record], FIELD_SPECS, arms={"closed": arm})
    await run_all([record], FIELD_SPECS, arms={"closed": arm})

    assert len(arm.calls) == 4


async def test_run_eval_limit_truncates_records(tmp_path: Path) -> None:
    lines = "\n".join(
        f'{{"item_id": "e{i}", "category": "prospect", "call_reason": "new_inquiry", '
        f'"variability_tier": "clean", "transcript": "t", "fields": '
        f'[{NAME_FIELD_RECORD!r}]}}'.replace("'", '"')
        for i in range(3)
    )
    (tmp_path / "eval.jsonl").write_text(lines + "\n", encoding="utf-8")

    arm = ClosedArm(chosen_index=0)
    results, categories = await run_eval(
        eval_dir=tmp_path, arms={"closed": arm}, concurrency=5, limit=1
    )

    assert categories == ["prospect"]
    assert len(results["closed"].datasets["prospect"].items) == 1
