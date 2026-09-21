"""Pydantic models for the category/field schema loaded at runtime from
`configs/categories/*.yaml` (see `trellis.schema.loader`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

MatchType = Literal[
    "exact",
    "normalized_phone",
    "fuzzy",
    "canonical_list",
    "numeric_tolerance",
    "date",
]


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    match_type: MatchType
    required: bool
    distractor_strategy: str
    value_pool: str | None = None


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_reasons: list[str]


class CategorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    fields: list[FieldSpec]
    scenarios: ScenarioSpec
