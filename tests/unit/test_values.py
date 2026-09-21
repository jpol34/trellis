from __future__ import annotations

import random
import re
from datetime import date

from faker import Faker

from trellis.generation.values import generate_value, load_value_pool
from trellis.schema.loader import CATEGORIES_DIR, load_all_categories

CATEGORIES = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}

PHONE_RE = re.compile(r"^\d{3}-\d{3}-\d{4}$")
EMAIL_RE = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _seeded_faker(seed: int) -> Faker:
    faker = Faker()
    faker.seed_instance(seed)
    return faker


def test_seeded_faker_and_rng_produce_deterministic_values():
    for spec in CATEGORIES.values():
        fields = spec.fields
        first = {f.name: generate_value(f, _seeded_faker(0), random.Random(0)) for f in fields}
        second = {f.name: generate_value(f, _seeded_faker(0), random.Random(0)) for f in fields}
        assert first == second


def test_phone_field_is_phone_shaped():
    field = next(f for f in CATEGORIES["prospect"].fields if f.name == "phone")
    for seed in range(10):
        value = generate_value(field, Faker(), random.Random(seed))
        assert PHONE_RE.match(value), value


def test_email_field_is_email_shaped():
    field = next(f for f in CATEGORIES["prospect"].fields if f.name == "email")
    for seed in range(10):
        value = generate_value(field, Faker(), random.Random(seed))
        assert EMAIL_RE.match(value), value


def test_canonical_list_field_is_member_of_its_value_pool():
    field = next(f for f in CATEGORIES["prospect"].fields if f.name == "bedroom_type")
    pool = load_value_pool("bedroom_types")
    for seed in range(10):
        value = generate_value(field, Faker(), random.Random(seed))
        assert value in pool


def test_numeric_tolerance_field_is_a_plausible_number():
    field = next(f for f in CATEGORIES["prospect"].fields if f.name == "budget")
    for seed in range(10):
        value = generate_value(field, Faker(), random.Random(seed))
        assert value.isdigit()
        assert 0 < int(value) < 100_000


def test_date_field_is_iso_date():
    field = next(f for f in CATEGORIES["prospect"].fields if f.name == "move_in_date")
    for seed in range(10):
        value = generate_value(field, Faker(), random.Random(seed))
        assert DATE_RE.match(value), value
        date.fromisoformat(value)  # must not raise


def test_fuzzy_field_returns_a_short_plausible_phrase():
    field = next(f for f in CATEGORIES["prospect"].fields if f.name == "tour_slot")
    for seed in range(10):
        value = generate_value(field, Faker(), random.Random(seed))
        assert isinstance(value, str)
        assert 0 < len(value) < 60


def test_every_field_in_every_category_produces_a_non_empty_value():
    for spec in CATEGORIES.values():
        for f in spec.fields:
            value = generate_value(f, Faker(), random.Random(42))
            assert value
