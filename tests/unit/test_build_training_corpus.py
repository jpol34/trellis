from __future__ import annotations

import itertools
import json
import random
from collections import Counter

from faker import Faker

from trellis.generation.build_training_corpus import build_training_corpus
from trellis.generation.corpus_common import VARIABILITY_TIERS
from trellis.generation.distractors import build_candidates
from trellis.generation.personas import Persona
from trellis.generation.values import generate_value
from trellis.schema.loader import CATEGORIES_DIR, load_all_categories

CATEGORIES = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}
N_CATEGORIES = len(CATEGORIES)


def _make_fake_generate_fn(seed: int = 0):
    """A stand-in for `generate_scenario_item` with the same (category, call_reason,
    variability_tier) -> dict signature, using real value/distractor generation (#3/#6-adjacent
    logic already covered by their own tests) but never making a network call."""
    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)
    counter = itertools.count()

    async def generate_fn(category, call_reason, variability_tier) -> dict:
        idx = next(counter)
        ground_truth = []
        for f in category.fields:
            value = generate_value(f, faker, rng)
            candidates, correct_index = build_candidates(value, f, faker, rng)
            ground_truth.append(
                {
                    "field": f.name,
                    "value": value,
                    "candidates": candidates,
                    "correct_index": correct_index,
                    "char_offset": None,
                }
            )
        transcript = (
            f"Caller: hi, calling about {call_reason}, item {idx}. Agent: got it, thanks."
        )
        persona = Persona(
            age_range="25-35", tone="neutral", verbosity="brief", background="n/a",
            speech_quirks="none",
        )
        return {"transcript": transcript, "persona": persona, "ground_truth": ground_truth}

    return generate_fn


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_build_training_corpus_produces_correct_total_and_ratio(tmp_path):
    total = 90  # divisible enough across cells/categories to hit exact 80/10/10 per cell
    counts = await build_training_corpus(
        total=total, out_dir=tmp_path, generate_fn=_make_fake_generate_fn()
    )

    assert sum(counts.values()) == total

    train = _read_jsonl(tmp_path / "train" / "records.jsonl")
    val = _read_jsonl(tmp_path / "val" / "records.jsonl")
    held_out = _read_jsonl(tmp_path / "held_out_internal" / "records.jsonl")

    assert len(train) == counts["train"]
    assert len(val) == counts["val"]
    assert len(held_out) == counts["held_out_internal"]
    assert len(train) + len(val) + len(held_out) == total

    # 80/10/10 per scenario cell, not just overall.
    cells = Counter()
    for split_name, split in (("train", train), ("val", val), ("held_out_internal", held_out)):
        for r in split:
            cells[(r["category"], r["call_reason"], r["variability_tier"], split_name)] += 1

    all_cells = {(c, cr, t) for (c, cr, t, _) in cells}
    for category, call_reason, tier in all_cells:
        n_train = cells[(category, call_reason, tier, "train")]
        n_val = cells[(category, call_reason, tier, "val")]
        n_held = cells[(category, call_reason, tier, "held_out_internal")]
        n_cell = n_train + n_val + n_held
        assert n_val == round(n_cell * 0.1)
        assert n_held == round(n_cell * 0.1)
        assert n_train == n_cell - n_val - n_held


async def test_build_training_corpus_record_shape(tmp_path):
    await build_training_corpus(total=10, out_dir=tmp_path, generate_fn=_make_fake_generate_fn())
    all_records = []
    for split in ("train", "val", "held_out_internal"):
        path = tmp_path / split / "records.jsonl"
        if path.exists():
            all_records += _read_jsonl(path)

    assert len(all_records) == 10
    for r in all_records:
        assert set(r) == {
            "item_id", "category", "call_reason", "variability_tier", "transcript", "persona",
            "fields",
        }
        assert r["category"] in CATEGORIES
        category = CATEGORIES[r["category"]]
        assert r["call_reason"] in category.scenarios.call_reasons
        assert r["variability_tier"] in VARIABILITY_TIERS
        assert isinstance(r["persona"], dict)
        assert {f["field"] for f in r["fields"]} == {f.name for f in category.fields}


async def test_build_training_corpus_reproducible_given_fixed_seed(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    await build_training_corpus(total=40, out_dir=out_a, generate_fn=_make_fake_generate_fn(seed=3))
    await build_training_corpus(total=40, out_dir=out_b, generate_fn=_make_fake_generate_fn(seed=3))

    for split in ("train", "val", "held_out_internal"):
        records_a = _read_jsonl(out_a / split / "records.jsonl")
        records_b = _read_jsonl(out_b / split / "records.jsonl")
        assert records_a == records_b


async def test_build_training_corpus_default_total_matches_400_per_category():
    from trellis.generation.build_training_corpus import TRANSCRIPTS_PER_CATEGORY

    assert TRANSCRIPTS_PER_CATEGORY == 400
