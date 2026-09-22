from __future__ import annotations

import pytest

from trellis.validate.states import resolve_state


def test_present_correct() -> None:
    assert resolve_state(True, False, True) == "present_correct"


def test_present_incorrect() -> None:
    assert resolve_state(True, False, False) == "present_incorrect"


def test_hallucinated() -> None:
    assert resolve_state(False, False, True) == "hallucinated"
    assert resolve_state(False, False, False) == "hallucinated"


def test_silently_dropped() -> None:
    assert resolve_state(True, True, None) == "silently_dropped"


def test_correctly_absent() -> None:
    assert resolve_state(False, True, None) == "correctly_absent"


def test_abstained_with_model_correct_set_raises() -> None:
    with pytest.raises(ValueError):
        resolve_state(True, True, True)


def test_not_abstained_with_model_correct_none_raises() -> None:
    with pytest.raises(ValueError):
        resolve_state(True, False, None)
