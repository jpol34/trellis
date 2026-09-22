from __future__ import annotations

import asyncio
import json
import random

import pytest

from trellis.generation.corpus_common import (
    VARIABILITY_TIERS,
    _load_checkpoint,
    allocate_cells_by_tier,
    allocate_cells_flat,
    allocate_counts,
    build_record,
    cells_flat,
    generate_records,
    record_to_quality_gate_item,
    stratified_split,
    write_jsonl,
)
from trellis.generation.personas import Persona
from trellis.generation.quality_gates import QualityGateItem
from trellis.schema.loader import CATEGORIES_DIR, load_all_categories

CATEGORIES = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}
PROSPECT = CATEGORIES["prospect"]
RESIDENT = CATEGORIES["resident"]

PERSONA = Persona(
    age_range="30-40", tone="calm", verbosity="brief", background="n/a", speech_quirks="none"
)


def _generated(call_reason: str) -> dict:
    return {
        "transcript": f"Caller: about {call_reason}. Agent: noted.",
        "persona": PERSONA,
        "ground_truth": [
            {
                "field": f.name,
                "value": f"value-for-{f.name}",
                "candidates": [f"value-for-{f.name}", "distractor-1", "distractor-2"],
                "correct_index": 0,
                "char_offset": None,
            }
            for f in PROSPECT.fields
        ],
    }


async def _fake_generate_fn(category, call_reason, variability_tier) -> dict:
    return _generated(call_reason)


def test_allocate_counts_sums_to_total_and_stays_within_one_of_base():
    rng = random.Random(0)
    cells = ["a", "b", "c", "d", "e", "f", "g"]
    counts = allocate_counts(cells, 20, rng)
    assert sum(counts.values()) == 20
    base = 20 // len(cells)
    assert all(c in (base, base + 1) for c in counts.values())


def test_allocate_counts_zero_total_gives_zero_everywhere():
    counts = allocate_counts(["a", "b"], 0, random.Random(0))
    assert counts == {"a": 0, "b": 0}


def test_allocate_counts_no_cells_nonzero_total_raises():
    with pytest.raises(ValueError):
        allocate_counts([], 5, random.Random(0))


def test_allocate_counts_no_cells_zero_total_is_empty():
    assert allocate_counts([], 0, random.Random(0)) == {}


def test_cells_flat_covers_every_call_reason_and_tier():
    cells = cells_flat(PROSPECT)
    assert len(cells) == len(PROSPECT.scenarios.call_reasons) * len(VARIABILITY_TIERS)
    assert set(cells) == {
        (cr, tier) for cr in PROSPECT.scenarios.call_reasons for tier in VARIABILITY_TIERS
    }


def test_allocate_cells_flat_sums_to_total():
    counts = allocate_cells_flat(PROSPECT, 50, random.Random(1))
    assert sum(counts.values()) == 50
    assert set(counts) == set(cells_flat(PROSPECT))


def test_allocate_cells_by_tier_gives_expected_split_for_50():
    counts = allocate_cells_by_tier(PROSPECT, 50, random.Random(1))
    assert sum(counts.values()) == 50
    per_tier: dict[str, int] = {tier: 0 for tier in VARIABILITY_TIERS}
    for (_, tier), n in counts.items():
        per_tier[tier] += n
    assert sorted(per_tier.values()) == [16, 17, 17]


def test_build_record_converts_persona_dataclass_and_carries_distractor_strategy():
    generated = _generated("new_inquiry")
    record = build_record(
        item_id="eval-prospect-new_inquiry-clean-0",
        category=PROSPECT,
        call_reason="new_inquiry",
        variability_tier="clean",
        generated=generated,
    )
    assert record["persona"] == {
        "age_range": "30-40",
        "tone": "calm",
        "verbosity": "brief",
        "background": "n/a",
        "speech_quirks": "none",
    }
    field_names = {f["field"] for f in record["fields"]}
    assert field_names == {f.name for f in PROSPECT.fields}
    name_field = next(f for f in record["fields"] if f["field"] == "name")
    assert name_field["distractor_strategy"] == "swap_similar_name"


def test_record_to_quality_gate_item_round_trips():
    generated = _generated("new_inquiry")
    record = build_record(
        item_id="item-0",
        category=PROSPECT,
        call_reason="new_inquiry",
        variability_tier="clean",
        generated=generated,
    )
    item = record_to_quality_gate_item(record)
    assert isinstance(item, QualityGateItem)
    assert item.item_id == "item-0"
    assert item.category == "prospect"
    assert len(item.fields) == len(record["fields"])


async def test_generate_records_produces_exactly_total_items():
    records = await generate_records(
        [PROSPECT, RESIDENT],
        60,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
    )
    assert len(records) == 60
    assert len({r["item_id"] for r in records}) == 60


async def test_generate_records_writes_checkpoint_incrementally(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    records = await generate_records(
        [PROSPECT],
        5,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
    )
    assert len(records) == 5
    checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) == 5
    assert {json.loads(line)["item_id"] for line in checkpoint_lines} == {
        r["item_id"] for r in records
    }


async def test_generate_records_resumes_from_checkpoint_without_regenerating(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    calls = {"n": 0}

    async def counting_generate_fn(category, call_reason, variability_tier) -> dict:
        calls["n"] += 1
        return _generated(call_reason)

    first_pass = await generate_records(
        [PROSPECT],
        5,
        generate_fn=counting_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
    )
    assert calls["n"] == 5

    second_pass = await generate_records(
        [PROSPECT],
        5,
        generate_fn=counting_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
    )
    # Same seed/total/id_prefix regenerates the identical set of item_ids, all already
    # checkpointed, so the (billed) generate_fn must not be called again.
    assert calls["n"] == 5
    assert {r["item_id"] for r in second_pass} == {r["item_id"] for r in first_pass}


async def test_generate_records_resumes_partial_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    calls = {"n": 0}

    async def failing_generate_fn(category, call_reason, variability_tier) -> dict:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated mid-run failure")
        return _generated(call_reason)

    with pytest.raises(RuntimeError):
        await generate_records(
            [PROSPECT],
            5,
            generate_fn=failing_generate_fn,
            rng=random.Random(0),
            id_prefix="t",
            checkpoint_path=checkpoint_path,
        )
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 2

    calls["n"] = 0

    async def succeeding_generate_fn(category, call_reason, variability_tier) -> dict:
        calls["n"] += 1
        return _generated(call_reason)

    records = await generate_records(
        [PROSPECT],
        5,
        generate_fn=succeeding_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
    )
    assert len(records) == 5
    # The 2 already-checkpointed items aren't regenerated on resume.
    assert calls["n"] == 3


async def test_generate_records_does_not_inflate_past_total_with_stale_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    await generate_records(
        [PROSPECT],
        20,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="eval",
        allocate_cells=allocate_cells_by_tier,
        checkpoint_path=checkpoint_path,
    )
    # Simulate the crash-before-cleanup window: a pilot run's checkpoint is left on disk, then a
    # later full run reuses the same out_dir/checkpoint path with a different (larger) total.
    records = await generate_records(
        [PROSPECT],
        100,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="eval",
        allocate_cells=allocate_cells_by_tier,
        checkpoint_path=checkpoint_path,
    )
    assert len(records) == 100
    assert len({r["item_id"] for r in records}) == 100


async def test_generate_records_does_not_inflate_past_a_smaller_total_with_stale_checkpoint(
    tmp_path,
):
    # The over-inclusion bug only manifested when a later run's total is *smaller* than what's
    # already on disk in the checkpoint (a larger stale checkpoint unconditionally seeded the
    # result before any per-item filtering happened) — a larger-total resume degenerates to the
    # same count either way and doesn't exercise the bug, so it's covered separately above.
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    await generate_records(
        [PROSPECT],
        100,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="eval",
        allocate_cells=allocate_cells_by_tier,
        checkpoint_path=checkpoint_path,
    )
    records = await generate_records(
        [PROSPECT],
        20,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="eval",
        allocate_cells=allocate_cells_by_tier,
        checkpoint_path=checkpoint_path,
    )
    assert len(records) == 20
    assert len({r["item_id"] for r in records}) == 20


def test_load_checkpoint_skips_a_truncated_trailing_line(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    good_record = _record("prospect", "new_inquiry", "clean", 0)
    checkpoint_path.write_text(
        json.dumps(good_record) + "\n" + '{"item_id": "prospect-new_inquiry-clean-1", "transcr',
        encoding="utf-8",
    )
    loaded = _load_checkpoint(checkpoint_path)
    assert list(loaded) == [good_record["item_id"]]


def test_load_checkpoint_skips_valid_json_missing_item_id(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    good_record = _record("prospect", "new_inquiry", "clean", 0)
    checkpoint_path.write_text(
        json.dumps(good_record) + "\n" + json.dumps({"foo": "bar"}) + "\n",
        encoding="utf-8",
    )
    loaded = _load_checkpoint(checkpoint_path)
    assert list(loaded) == [good_record["item_id"]]


async def test_generate_records_only_uses_known_cells():
    records = await generate_records(
        [PROSPECT], 15, generate_fn=_fake_generate_fn, rng=random.Random(0), id_prefix="t"
    )
    for r in records:
        assert r["call_reason"] in PROSPECT.scenarios.call_reasons
        assert r["variability_tier"] in VARIABILITY_TIERS


async def test_generate_records_concurrency_overlaps_calls(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    in_flight = {"current": 0, "max": 0}

    async def slow_generate_fn(category, call_reason, variability_tier) -> dict:
        in_flight["current"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["current"])
        await asyncio.sleep(0.05)
        in_flight["current"] -= 1
        return _generated(call_reason)

    records = await generate_records(
        [PROSPECT],
        10,
        generate_fn=slow_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
        concurrency=5,
    )
    assert len(records) == 10
    # With concurrency=5, at least some calls must have been in flight simultaneously — strictly
    # sequential execution (as with the concurrency=1 default) would never exceed 1.
    assert in_flight["max"] > 1


async def test_generate_records_concurrency_checkpoint_has_no_corruption_or_duplicates(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    records = await generate_records(
        [PROSPECT, RESIDENT],
        40,
        generate_fn=_fake_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
        concurrency=8,
    )
    assert len(records) == 40
    assert len({r["item_id"] for r in records}) == 40

    checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) == 40
    checkpoint_ids = [json.loads(line)["item_id"] for line in checkpoint_lines]
    assert len(checkpoint_ids) == len(set(checkpoint_ids)) == 40


async def test_generate_records_concurrency_resumes_from_checkpoint_without_regenerating(
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    calls = {"n": 0}

    async def counting_generate_fn(category, call_reason, variability_tier) -> dict:
        calls["n"] += 1
        return _generated(call_reason)

    first_pass = await generate_records(
        [PROSPECT],
        12,
        generate_fn=counting_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
        concurrency=4,
    )
    assert calls["n"] == 12

    second_pass = await generate_records(
        [PROSPECT],
        12,
        generate_fn=counting_generate_fn,
        rng=random.Random(0),
        id_prefix="t",
        checkpoint_path=checkpoint_path,
        concurrency=4,
    )
    assert calls["n"] == 12
    assert {r["item_id"] for r in second_pass} == {r["item_id"] for r in first_pass}


async def test_generate_records_concurrency_fails_fast_with_original_exception_type(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    class BoomError(RuntimeError):
        pass

    async def flaky_generate_fn(category, call_reason, variability_tier) -> dict:
        if call_reason == PROSPECT.scenarios.call_reasons[0]:
            await asyncio.sleep(0.05)
            raise BoomError("simulated concurrent failure")
        await asyncio.sleep(0.1)
        return _generated(call_reason)

    with pytest.raises(BoomError):
        await generate_records(
            [PROSPECT],
            10,
            generate_fn=flaky_generate_fn,
            rng=random.Random(0),
            id_prefix="t",
            checkpoint_path=checkpoint_path,
            concurrency=5,
        )
    # Sibling tasks still in flight when the failure hit must have been cancelled rather than
    # left to complete, so the checkpoint holds strictly fewer than the full total.
    checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) < 10


def _record(category: str, call_reason: str, tier: str, idx: int) -> dict:
    return {
        "item_id": f"{category}-{call_reason}-{tier}-{idx}",
        "category": category,
        "call_reason": call_reason,
        "variability_tier": tier,
        "transcript": f"transcript {idx}",
        "persona": {},
        "fields": [],
    }


def test_stratified_split_is_per_cell_not_pooled():
    # A small cell (n=10) alongside a much bigger one (n=100) — splitting the whole pool as one
    # 110-item batch would leave the small cell's per-split counts to chance; splitting per cell
    # gives it its own proportional 8/1/1 split regardless of the big cell's size.
    records = [_record("prospect", "small_reason", "clean", i) for i in range(10)]
    records += [_record("prospect", "big_reason", "clean", i) for i in range(100)]

    train, val, held_out = stratified_split(records, seed=42)

    small_cell_splits = [
        sum(1 for r in split if r["call_reason"] == "small_reason")
        for split in (train, val, held_out)
    ]
    assert small_cell_splits == [8, 1, 1]

    big_cell_splits = [
        sum(1 for r in split if r["call_reason"] == "big_reason")
        for split in (train, val, held_out)
    ]
    assert big_cell_splits == [80, 10, 10]
    assert sum(len(s) for s in (train, val, held_out)) == len(records)


def test_stratified_split_reproducible_given_fixed_seed():
    records = [_record("prospect", "new_inquiry", "clean", i) for i in range(20)]
    split_a = stratified_split(records, seed=7)
    split_b = stratified_split(records, seed=7)
    assert [[r["item_id"] for r in s] for s in split_a] == [
        [r["item_id"] for r in s] for s in split_b
    ]


def test_stratified_split_no_item_duplicated_or_dropped():
    records = [
        _record("prospect", "new_inquiry", tier, i)
        for tier in VARIABILITY_TIERS
        for i in range(7)
    ]
    train, val, held_out = stratified_split(records, seed=1)
    all_ids = [r["item_id"] for r in (*train, *val, *held_out)]
    assert sorted(all_ids) == sorted(r["item_id"] for r in records)


def test_write_jsonl_round_trips(tmp_path):
    records = [_record("prospect", "new_inquiry", "clean", i) for i in range(3)]
    path = tmp_path / "sub" / "out.jsonl"
    n = write_jsonl(path, records)
    assert n == 3
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["item_id"] for line in lines] == [r["item_id"] for r in records]
