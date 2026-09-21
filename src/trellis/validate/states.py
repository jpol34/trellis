"""The 5-state resolution both scoring modes (closed_set.py and matchers.py-based open
extraction) produce, so downstream metrics (metrics.py) work identically regardless of which
arm generated the prediction."""

from __future__ import annotations

from typing import Literal

FiveState = Literal[
    "present_correct",
    "present_incorrect",
    "hallucinated",
    "silently_dropped",
    "correctly_absent",
]


def resolve_state(
    field_required_and_stated_in_gold: bool,
    model_abstained: bool,
    model_correct: bool | None,
) -> FiveState:
    """Resolves a single field's scoring outcome into one of the 5 states.

    `field_required_and_stated_in_gold`: whether the gold transcript actually contains a value
    for this field (i.e. the field is NOT correctly-absent by ground truth).
    `model_abstained`: whether the model declined to state a value (chose "not mentioned" /
    returned no text).
    `model_correct`: for a non-abstaining model, whether its stated value matched gold. Must be
    `None` when `model_abstained` is True (abstention has no correctness to report) and must be
    a `bool` otherwise.
    """
    if model_abstained:
        if model_correct is not None:
            raise ValueError("model_correct must be None when model_abstained is True")
        return "correctly_absent" if not field_required_and_stated_in_gold else "silently_dropped"

    if model_correct is None:
        raise ValueError("model_correct must be a bool when model_abstained is False")

    if not field_required_and_stated_in_gold:
        return "hallucinated"

    return "present_correct" if model_correct else "present_incorrect"
