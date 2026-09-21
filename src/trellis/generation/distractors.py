"""Plausible-but-wrong candidate generation for injected ground-truth values.

Each field declares a `distractor_strategy` (see #3's `FieldSpec`); `_STRATEGIES` maps that
string to a generator producing exactly 3 distractors sharing the real value's domain/format
(e.g. a phone distractor keeps the real number's area code; a canonical_list distractor is
another member of the same value pool). `build_candidates` assembles the full, shuffled
`candidates: list[str]` + `correct_index` pair — the shape #4's wire adapter
(`candidates_to_criteria`) consumes — adding a "not mentioned" option for non-required fields.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, timedelta

from faker import Faker

from trellis.generation.values import fuzzy_phrase_pool, load_value_pool, numeric_range
from trellis.schema.types import FieldSpec

NOT_MENTIONED = "not mentioned"

_DistractorFn = Callable[[str, FieldSpec, Faker, random.Random], list[str]]
_STRATEGIES: dict[str, _DistractorFn] = {}


def _strategy(name: str) -> Callable[[_DistractorFn], _DistractorFn]:
    def register(fn: _DistractorFn) -> _DistractorFn:
        _STRATEGIES[name] = fn
        return fn

    return register


@_strategy("swap_similar_name")
def _swap_similar_name(value: str, field: FieldSpec, faker: Faker, rng: random.Random) -> list[str]:
    out: list[str] = []
    seen = {value}
    while len(out) < 3:
        candidate = faker.name()
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


@_strategy("perturb_digit")
def _perturb_digit(value: str, field: FieldSpec, faker: Faker, rng: random.Random) -> list[str]:
    area = value.split("-", 1)[0]
    out: list[str] = []
    seen = {value}
    while len(out) < 3:
        candidate = f"{area}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


@_strategy("swap_domain")
def _swap_domain(value: str, field: FieldSpec, faker: Faker, rng: random.Random) -> list[str]:
    local = value.split("@", 1)[0]
    out: list[str] = []
    seen = {value}
    while len(out) < 3:
        candidate = f"{local}@{faker.free_email_domain()}"
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


@_strategy("perturb_amount")
def _perturb_amount(value: str, field: FieldSpec, faker: Faker, rng: random.Random) -> list[str]:
    base = float(value)
    low, high = numeric_range(field)
    out: list[str] = []
    seen = {value}
    attempts = 0
    while len(out) < 3 and attempts < 100:
        attempts += 1
        delta = base * rng.uniform(0.1, 0.4) * rng.choice([-1, 1])
        candidate_val = min(max(round(base + delta), low), high)
        candidate = str(int(candidate_val))
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    if len(out) < 3:
        raise ValueError(f"could not generate 3 distinct amount distractors near {value!r}")
    return out


@_strategy("shift_date")
def _shift_date(value: str, field: FieldSpec, faker: Faker, rng: random.Random) -> list[str]:
    base = date.fromisoformat(value)
    offsets = [d for d in range(-21, 22) if d != 0]
    rng.shuffle(offsets)
    out: list[str] = []
    seen = {value}
    for offset in offsets:
        candidate = (base + timedelta(days=offset)).isoformat()
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
        if len(out) == 3:
            break
    return out


@_strategy("swap_sibling_value")
def _swap_sibling_value(
    value: str, field: FieldSpec, faker: Faker, rng: random.Random
) -> list[str]:
    assert field.value_pool is not None
    pool = [v for v in load_value_pool(field.value_pool) if v != value]
    if len(pool) < 3:
        raise ValueError(
            f"value pool {field.value_pool!r} has too few sibling values to produce 3 "
            f"distractors distinct from {value!r}"
        )
    return rng.sample(pool, 3)


@_strategy("shift_window")
def _shift_window(value: str, field: FieldSpec, faker: Faker, rng: random.Random) -> list[str]:
    pool = [p for p in fuzzy_phrase_pool(field.name) if p != value]
    if len(pool) < 3:
        raise ValueError(
            f"fuzzy phrase pool for {field.name!r} has too few options to produce 3 distractors "
            f"distinct from {value!r}"
        )
    return rng.sample(pool, 3)


@_strategy("swap_adjacent_unit")
def _swap_adjacent_unit(
    value: str, field: FieldSpec, faker: Faker, rng: random.Random
) -> list[str]:
    n = int(value)
    out: list[str] = []
    seen = {value}
    deltas = [1, -1, 2, -2, 3, -3, 10, -10]
    rng.shuffle(deltas)
    for delta in deltas:
        candidate_n = n + delta
        if candidate_n <= 0:
            continue
        candidate = str(candidate_n)
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
        if len(out) == 3:
            break
    return out


def generate_distractors(
    value: str, field: FieldSpec, faker: Faker, rng: random.Random
) -> list[str]:
    strategy = _STRATEGIES.get(field.distractor_strategy)
    if strategy is None:
        raise ValueError(f"no distractor strategy registered for {field.distractor_strategy!r}")
    distractors = strategy(value, field, faker, rng)
    if len(distractors) != 3 or len(set(distractors)) != 3 or value in distractors:
        raise ValueError(
            f"strategy {field.distractor_strategy!r} for field {field.name!r} did not produce "
            f"exactly 3 distinct distractors distinct from the real value (got {distractors!r})"
        )
    return distractors


def build_candidates(
    value: str, field: FieldSpec, faker: Faker, rng: random.Random
) -> tuple[list[str], int]:
    """Builds the shuffled `candidates: list[str]` + `correct_index` pair for one field: the
    real value plus its 3 distractors, with a "not mentioned" option appended for non-required
    fields. Shuffle order is randomized per call (via `rng`) to avoid position bias."""
    candidates = [value, *generate_distractors(value, field, faker, rng)]
    if not field.required:
        candidates.append(NOT_MENTIONED)
    rng.shuffle(candidates)
    return candidates, candidates.index(value)
