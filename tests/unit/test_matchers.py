"""Covers all 6 match_types for the open-extraction (GPT-5.1) scoring path: each matcher feeds
into `states.resolve_state` the same way the real arm's abstention/value output would, so both
scoring modes are exercised against the same 5-state space as closed_set.py."""

from __future__ import annotations

from trellis.validate.matchers import (
    canonical_list_match,
    date_match,
    exact_match,
    fuzzy_match,
    normalized_phone_match,
    numeric_tolerance_match,
)
from trellis.validate.states import resolve_state


def _open_state(field_required: bool, abstained: bool, matcher_correct: bool | None):
    return resolve_state(field_required, abstained, matcher_correct)


# --- exact (email) ---


def test_exact_present_correct() -> None:
    correct = exact_match("jane@acme.com", "jane@acme.com")
    assert _open_state(True, False, correct) == "present_correct"


def test_exact_present_incorrect() -> None:
    correct = exact_match("john@acme.com", "jane@acme.com")
    assert _open_state(True, False, correct) == "present_incorrect"


def test_exact_hallucinated() -> None:
    # field genuinely absent from transcript, but the model states a value anyway
    correct = exact_match("ghost@acme.com", "ghost@acme.com")
    assert _open_state(False, False, correct) == "hallucinated"


def test_exact_silently_dropped() -> None:
    assert _open_state(True, True, None) == "silently_dropped"


def test_exact_correctly_absent() -> None:
    assert _open_state(False, True, None) == "correctly_absent"


# --- normalized_phone ---


def test_phone_present_correct() -> None:
    correct = normalized_phone_match("(212) 555-0123", "212-555-0123")
    assert correct is True
    assert _open_state(True, False, correct) == "present_correct"


def test_phone_present_incorrect() -> None:
    correct = normalized_phone_match("(212) 555-0199", "212-555-0123")
    assert correct is False
    assert _open_state(True, False, correct) == "present_incorrect"


def test_phone_hallucinated() -> None:
    correct = normalized_phone_match("212-555-0199", "212-555-0199")
    assert _open_state(False, False, correct) == "hallucinated"


def test_phone_silently_dropped() -> None:
    assert _open_state(True, True, None) == "silently_dropped"


def test_phone_correctly_absent() -> None:
    assert _open_state(False, True, None) == "correctly_absent"


# --- fuzzy (name, default threshold 0.85) ---


def test_fuzzy_present_correct() -> None:
    correct = fuzzy_match("Katherine Lee", "Katherine Lee")
    assert correct is True
    assert _open_state(True, False, correct) == "present_correct"


def test_fuzzy_present_incorrect() -> None:
    correct = fuzzy_match("Robert Jones", "Katherine Lee")
    assert correct is False
    assert _open_state(True, False, correct) == "present_incorrect"


def test_fuzzy_hallucinated() -> None:
    correct = fuzzy_match("Ghost Person", "Ghost Person")
    assert _open_state(False, False, correct) == "hallucinated"


def test_fuzzy_silently_dropped() -> None:
    assert _open_state(True, True, None) == "silently_dropped"


def test_fuzzy_correctly_absent() -> None:
    assert _open_state(False, True, None) == "correctly_absent"


# --- canonical_list (bedroom_type) ---

BEDROOM_POOL = ["studio", "1br", "2br", "3br"]


def test_canonical_list_present_correct() -> None:
    correct = canonical_list_match("2br", "2br", BEDROOM_POOL)
    assert correct is True
    assert _open_state(True, False, correct) == "present_correct"


def test_canonical_list_present_incorrect() -> None:
    correct = canonical_list_match("1br", "2br", BEDROOM_POOL)
    assert correct is False
    assert _open_state(True, False, correct) == "present_incorrect"


def test_canonical_list_rejects_value_outside_pool() -> None:
    # "2 bedroom" reads the same as gold's "2br" but isn't a literal pool member, so it's
    # rejected outright rather than credited as a match.
    assert canonical_list_match("2 bedroom", "2br", BEDROOM_POOL) is False


def test_canonical_list_hallucinated() -> None:
    correct = canonical_list_match("3br", "3br", BEDROOM_POOL)
    assert _open_state(False, False, correct) == "hallucinated"


def test_canonical_list_silently_dropped() -> None:
    assert _open_state(True, True, None) == "silently_dropped"


def test_canonical_list_correctly_absent() -> None:
    assert _open_state(False, True, None) == "correctly_absent"


# --- numeric_tolerance (balance_owed) ---


def test_numeric_present_correct() -> None:
    correct = numeric_tolerance_match("$1,200.00", "1200", tolerance=0.5)
    assert correct is True
    assert _open_state(True, False, correct) == "present_correct"


def test_numeric_present_incorrect() -> None:
    correct = numeric_tolerance_match("$500", "1200", tolerance=0.5)
    assert correct is False
    assert _open_state(True, False, correct) == "present_incorrect"


def test_numeric_hallucinated() -> None:
    correct = numeric_tolerance_match("$100", "100", tolerance=0.5)
    assert _open_state(False, False, correct) == "hallucinated"


def test_numeric_silently_dropped() -> None:
    assert _open_state(True, True, None) == "silently_dropped"


def test_numeric_correctly_absent() -> None:
    assert _open_state(False, True, None) == "correctly_absent"


# --- date (lease_renewal_date) ---


def test_date_present_correct() -> None:
    correct = date_match("2026-09-21", "September 21, 2026")
    assert correct is True
    assert _open_state(True, False, correct) == "present_correct"


def test_date_present_incorrect() -> None:
    correct = date_match("2026-09-22", "September 21, 2026")
    assert correct is False
    assert _open_state(True, False, correct) == "present_incorrect"


def test_date_hallucinated() -> None:
    correct = date_match("2026-01-01", "2026-01-01")
    assert _open_state(False, False, correct) == "hallucinated"


def test_date_silently_dropped() -> None:
    assert _open_state(True, True, None) == "silently_dropped"


def test_date_correctly_absent() -> None:
    assert _open_state(False, True, None) == "correctly_absent"


def test_date_unparseable_does_not_match() -> None:
    assert date_match("not a date", "2026-01-01") is False
