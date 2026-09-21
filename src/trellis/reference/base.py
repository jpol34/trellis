"""Adapter interface and registry for reference arms (the models being
benchmarked against each other on a field-extraction question).

Each arm module registers itself into `ARMS` by name at import time.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from trellis.schema.types import FieldSpec


class ArmAnswer(TypedDict):
    value: str | None
    chosen_index: int | None
    confidence: float


class ReferenceArm(Protocol):
    name: str

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
