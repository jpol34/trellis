"""Per-`match_type` matchers for the GPT-5.1 open-extraction arm, which returns free text
rather than a candidate index (unlike the closed-set arms scored by `closed_set.py`). Each
matcher answers "does `predicted` match `gold`" for its `FieldSpec.match_type`; the caller is
responsible for feeding the result into `states.resolve_state` alongside abstention detection."""

from __future__ import annotations

import re
from datetime import date, datetime

import phonenumbers
from rapidfuzz import fuzz


def exact_match(predicted: str, gold: str) -> bool:
    return predicted.strip().casefold() == gold.strip().casefold()


def normalized_phone_match(predicted: str, gold: str) -> bool:
    try:
        predicted_number = phonenumbers.parse(predicted, "US")
        gold_number = phonenumbers.parse(gold, "US")
    except phonenumbers.NumberParseException:
        return False
    if not (
        phonenumbers.is_valid_number(predicted_number)
        and phonenumbers.is_valid_number(gold_number)
    ):
        return False
    return phonenumbers.national_significant_number(
        predicted_number
    ) == phonenumbers.national_significant_number(gold_number)


def fuzzy_match(predicted: str, gold: str, threshold: float = 0.85) -> bool:
    score = fuzz.ratio(predicted.strip().casefold(), gold.strip().casefold()) / 100.0
    return score >= threshold


def canonical_list_match(predicted: str, gold: str, value_pool: list[str]) -> bool:
    pool_casefold = {v.strip().casefold() for v in value_pool}
    predicted_norm = predicted.strip().casefold()
    if predicted_norm not in pool_casefold:
        return False
    return predicted_norm == gold.strip().casefold()


_CURRENCY_STRIP = re.compile(r"[$,\s]")
_PARENTHESIZED_NEGATIVE = re.compile(r"^\((.+)\)$")


def _parse_amount(value: str) -> float:
    stripped = _CURRENCY_STRIP.sub("", value.strip())
    # Accounting notation writes a negative amount as "(500)" rather than "-500".
    match = _PARENTHESIZED_NEGATIVE.match(stripped)
    if match:
        stripped = f"-{match.group(1)}"
    return float(stripped)


def numeric_tolerance_match(predicted: str, gold: str, tolerance: float) -> bool:
    try:
        predicted_num = _parse_amount(predicted)
        gold_num = _parse_amount(gold)
    except ValueError:
        return False
    return abs(predicted_num - gold_num) <= tolerance


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y")


def _parse_date(value: str) -> date | None:
    value = value.strip()
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def date_match(predicted: str, gold: str) -> bool:
    predicted_date = _parse_date(predicted)
    gold_date = _parse_date(gold)
    if predicted_date is None or gold_date is None:
        return False
    return predicted_date == gold_date
