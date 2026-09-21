"""Synthetic ground-truth value generation for one scenario cell.

Uses a seeded `Faker` for name/phone/email, curated value-pool sampling (via the schema loader)
for `canonical_list` fields, and reasonable synthetic generation for the remaining field types:
a plausible domain-appropriate number for `numeric_tolerance`, a plausible near-future/near-past
date for `date`, and a short plausible phrase for fuzzy free-text fields.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import yaml
from faker import Faker

from trellis.schema.loader import VALUE_POOLS_DIR
from trellis.schema.types import FieldSpec

# Domain-appropriate numeric ranges keyed by field name — a plain "10-5000" fallback would let a
# budget and a balance_owed land in the same implausible range as each other.
_NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "budget": (900, 4500),
    "balance_owed": (0, 3200),
}
_DEFAULT_NUMERIC_RANGE = (10, 5000)

_FUZZY_PHRASE_POOLS: dict[str, list[str]] = {
    "tour_slot": [
        "tomorrow afternoon around 2",
        "Saturday morning",
        "Tuesday after 5pm",
        "this weekend, whenever works",
        "next Monday at 10am",
        "Thursday around lunchtime",
    ],
    "callback_window": [
        "anytime after 6pm",
        "weekday mornings",
        "lunchtime, around noon",
        "before 9am",
        "Friday afternoon",
        "whenever, I work from home",
    ],
}
_DEFAULT_FUZZY_PHRASE_POOL = [
    "sometime this week",
    "whenever is convenient",
    "no strong preference",
]


def load_value_pool(name: str) -> list[str]:
    path = VALUE_POOLS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fuzzy_phrase_pool(field_name: str) -> list[str]:
    """The candidate phrase pool for a fuzzy free-text field, shared between value generation
    and `distractors.py` so a distractor is drawn from the same pool as the real value."""
    return _FUZZY_PHRASE_POOLS.get(field_name, _DEFAULT_FUZZY_PHRASE_POOL)


def sample_phone(rng: random.Random) -> str:
    area = rng.randint(200, 999)
    exchange = rng.randint(200, 999)
    line = rng.randint(1000, 9999)
    return f"{area}-{exchange}-{line}"


def numeric_range(field: FieldSpec) -> tuple[int, int]:
    return _NUMERIC_RANGES.get(field.name, _DEFAULT_NUMERIC_RANGE)


def _sample_numeric(field: FieldSpec, rng: random.Random) -> str:
    low, high = numeric_range(field)
    return str(rng.randint(low, high))


def _sample_date(rng: random.Random) -> str:
    offset = rng.randint(-30, 90)
    return (date.today() + timedelta(days=offset)).isoformat()


def _sample_fuzzy(field: FieldSpec, rng: random.Random) -> str:
    return rng.choice(fuzzy_phrase_pool(field.name))


def generate_value(field: FieldSpec, faker: Faker, rng: random.Random) -> str:
    """Generates a plausible ground-truth value for `field`, dispatched primarily on field name
    (email/unit_number/name all share `match_type` with other fields) and falling back to
    `match_type` for the rest."""
    if field.name == "name":
        return faker.name()
    if field.match_type == "normalized_phone":
        return sample_phone(rng)
    if field.name == "email":
        return faker.email()
    if field.match_type == "canonical_list":
        assert field.value_pool is not None
        return rng.choice(load_value_pool(field.value_pool))
    if field.match_type == "numeric_tolerance":
        return _sample_numeric(field, rng)
    if field.match_type == "date":
        return _sample_date(rng)
    if field.name == "unit_number":
        return str(rng.randint(101, 450))
    return _sample_fuzzy(field, rng)
