"""Discovers every category definition under `configs/categories/`, validates each against
`CategorySpec`, and cross-checks `canonical_list` field `value_pool` references against
`configs/value_pools/`, so a malformed or dangling config fails fast at load time rather than
surfacing as an opaque error mid-generation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from trellis.schema.types import CategorySpec

_CATEGORY_ADAPTER: TypeAdapter[CategorySpec] = TypeAdapter(CategorySpec)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATEGORIES_DIR = REPO_ROOT / "configs" / "categories"
VALUE_POOLS_DIR = REPO_ROOT / "configs" / "value_pools"


class SchemaError(ValueError):
    """A category or value-pool config failed validation."""


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SchemaError(f"{path}: invalid YAML\n{e}") from e


def _load_value_pool_names(value_pools_dir: Path) -> set[str]:
    names = set()
    for path in value_pools_dir.glob("*.yaml"):
        raw = _load_yaml(path)
        if raw:
            names.add(path.stem)
    return names


def load_category(path: Path, value_pools_dir: Path = VALUE_POOLS_DIR) -> CategorySpec:
    raw = _load_yaml(path)
    if raw is None:
        raise SchemaError(f"{path}: file is empty")

    try:
        spec = _CATEGORY_ADAPTER.validate_python(raw)
    except ValidationError as e:
        raise SchemaError(f"{path}: invalid category definition\n{e}") from e

    available_pools = _load_value_pool_names(value_pools_dir)
    for field in spec.fields:
        if field.match_type == "canonical_list" and not field.value_pool:
            raise SchemaError(
                f"{path}: field {field.name!r} has match_type 'canonical_list' but no "
                "value_pool reference"
            )
        if field.value_pool and field.value_pool not in available_pools:
            raise SchemaError(
                f"{path}: field {field.name!r} references value_pool "
                f"{field.value_pool!r}, which does not resolve to a real, non-empty "
                f"file under {value_pools_dir}"
            )

    return spec


def load_all_categories(directory: Path = CATEGORIES_DIR) -> list[CategorySpec]:
    return [load_category(p) for p in sorted(directory.glob("*.yaml"))]
