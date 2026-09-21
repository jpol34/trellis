from __future__ import annotations

import random
import re

import pytest
from faker import Faker

from trellis.generation.distractors import (
    NOT_MENTIONED,
    build_candidates,
    generate_distractors,
)
from trellis.generation.values import load_value_pool
from trellis.schema.loader import CATEGORIES_DIR, load_all_categories

CATEGORIES = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}
PROSPECT = CATEGORIES["prospect"]
RESIDENT = CATEGORIES["resident"]

PHONE_RE = re.compile(r"^\d{3}-\d{3}-\d{4}$")
EMAIL_RE = re.compile(r"^[^@]+@[^@]+\.[^@]+$")


def _field(spec, name):
    return next(f for f in spec.fields if f.name == name)


def _rng_faker(seed: int) -> tuple[random.Random, Faker]:
    return random.Random(seed), Faker()


def test_swap_similar_name_produces_three_distinct_names_excluding_real():
    field = _field(PROSPECT, "name")
    rng, faker = _rng_faker(1)
    distractors = generate_distractors("Jamie Rivera", field, faker, rng)
    assert len(distractors) == 3
    assert len(set(distractors)) == 3
    assert "Jamie Rivera" not in distractors


def test_perturb_digit_keeps_area_code_and_is_phone_shaped():
    field = _field(PROSPECT, "phone")
    rng, faker = _rng_faker(2)
    real = "555-123-4567"
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert PHONE_RE.match(d), d
        assert d.split("-")[0] == "555"
    assert real not in distractors


def test_swap_domain_keeps_local_part_and_is_email_shaped():
    field = _field(PROSPECT, "email")
    rng, faker = _rng_faker(3)
    real = "jamie.rivera@example.com"
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert EMAIL_RE.match(d), d
        assert d.split("@")[0] == "jamie.rivera"
    assert real not in distractors


def test_perturb_amount_is_numeric_and_near_real_value():
    field = _field(PROSPECT, "budget")
    rng, faker = _rng_faker(4)
    real = "2000"
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert d.isdigit()
        assert d != real
        assert abs(int(d) - int(real)) <= int(real) * 0.5 + 1


def test_perturb_amount_handles_zero_base_value():
    field = _field(RESIDENT, "balance_owed")
    rng, faker = _rng_faker(6)
    real = "0"
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert d.isdigit()
        assert d != real


def test_shift_date_is_iso_date_near_real_value():
    field = _field(PROSPECT, "move_in_date")
    rng, faker = _rng_faker(5)
    real = "2026-10-01"
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", d)
        assert d != real


def test_swap_sibling_value_is_member_of_same_pool_minus_real():
    field = _field(PROSPECT, "bedroom_type")
    rng, faker = _rng_faker(6)
    pool = load_value_pool("bedroom_types")
    real = pool[0]
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert d in pool
        assert d != real


def test_shift_window_is_member_of_same_fuzzy_pool():
    field = _field(PROSPECT, "tour_slot")
    rng, faker = _rng_faker(7)
    from trellis.generation.values import fuzzy_phrase_pool

    pool = fuzzy_phrase_pool("tour_slot")
    real = pool[0]
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert d in pool
        assert d != real


def test_swap_adjacent_unit_is_numeric_and_near_real_value():
    field = _field(RESIDENT, "unit_number")
    rng, faker = _rng_faker(8)
    real = "204"
    distractors = generate_distractors(real, field, faker, rng)
    assert len(distractors) == 3
    for d in distractors:
        assert d.isdigit()
        assert d != real
        assert abs(int(d) - int(real)) <= 10


def test_unknown_strategy_raises():
    from trellis.schema.types import FieldSpec

    field = FieldSpec(
        name="mystery",
        match_type="exact",
        required=True,
        distractor_strategy="does_not_exist",
    )
    rng, faker = _rng_faker(9)
    with pytest.raises(ValueError):
        generate_distractors("value", field, faker, rng)


def test_build_candidates_required_field_has_four_options_no_not_mentioned():
    field = _field(PROSPECT, "bedroom_type")
    rng, faker = _rng_faker(10)
    pool = load_value_pool("bedroom_types")
    real = pool[1]
    candidates, correct_index = build_candidates(real, field, faker, rng)
    assert len(candidates) == 4
    assert NOT_MENTIONED not in candidates
    assert candidates[correct_index] == real
    assert len(set(candidates)) == 4


def test_build_candidates_optional_field_has_five_options_with_not_mentioned():
    field = _field(PROSPECT, "pet_info")
    assert field.required is False
    rng, faker = _rng_faker(11)
    pool = load_value_pool("pet_policies")
    real = pool[0]
    candidates, correct_index = build_candidates(real, field, faker, rng)
    assert len(candidates) == 5
    assert NOT_MENTIONED in candidates
    assert candidates[correct_index] == real


def test_build_candidates_shuffle_order_varies_across_calls():
    field = _field(PROSPECT, "bedroom_type")
    faker = Faker()
    pool = load_value_pool("bedroom_types")
    real = pool[1]

    positions = set()
    for seed in range(20):
        candidates, correct_index = build_candidates(real, field, faker, random.Random(seed))
        positions.add(correct_index)

    assert len(positions) > 1


def test_all_field_distractor_strategies_are_registered():
    for spec in (PROSPECT, RESIDENT):
        rng, faker = _rng_faker(hash(spec.category) & 0xFFFF)
        for f in spec.fields:
            from trellis.generation.values import generate_value

            value = generate_value(f, faker, rng)
            distractors = generate_distractors(value, f, faker, rng)
            assert len(distractors) == 3
