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
