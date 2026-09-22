"""Adapter interface and registry for reference arms (the models being
benchmarked against each other on a field-extraction question).

Each arm module registers itself into `ARMS` by name at import time.
"""

from __future__ import annotations

from typing import NotRequired, Protocol, TypedDict

from trellis.schema.types import FieldSpec


class ArmAnswer(TypedDict):
    value: str | None
    chosen_index: int | None
    confidence: float
    # Real token usage from the arm's own API response, when it has one (gpt-5.1, jev) — used to
    # compute real $ cost in the eval report (validate/costs.py). Absent/None for trellis, which
    # has no token concept; NotRequired so every existing ArmAnswer construction keeps compiling.
    input_tokens: NotRequired[int | None]
    output_tokens: NotRequired[int | None]


class ReferenceArm(Protocol):
    name: str
    # "closed_set": `answer()` is always given `candidates` and scored by `validate/closed_set.py`.
    # "open_extraction": `answer()` is always given `candidates=None` and scored by
    # `validate/matchers.py`.
    mode: str
    # Optional: caps how many `answer()` calls the eval runner keeps in flight for this arm at
    # once, overriding the runner's own `--concurrency`. Arms that internally serialize (e.g. one
    # shared model instance behind a lock) should set this below the runner default so queueing
    # time isn't misreported as inference latency in the rendered report.
    max_concurrency: int

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer: ...


ARMS: dict[str, ReferenceArm] = {}


def register_arm(arm: ReferenceArm) -> None:
    """Registers `arm` into `ARMS` under `arm.name`. Raises `ValueError` if that name is
    already registered, so two arms colliding on a name fail loudly instead of one silently
    overwriting the other."""
    if arm.name in ARMS:
        raise ValueError(f"a reference arm named {arm.name!r} is already registered")
    ARMS[arm.name] = arm
