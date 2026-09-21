from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trellis.schema.loader import (
    CATEGORIES_DIR,
    VALUE_POOLS_DIR,
    SchemaError,
    load_all_categories,
    load_category,
)

RESIDENT_YAML = CATEGORIES_DIR / "resident.yaml"
PROSPECT_YAML = CATEGORIES_DIR / "prospect.yaml"


def test_resident_loads_and_validates() -> None:
    spec = load_category(RESIDENT_YAML)
    assert spec.category == "resident"
    assert {f.name for f in spec.fields} == {
        "name",
        "phone",
        "email",
        "unit_number",
        "issue_type",
        "issue_severity",
        "callback_window",
        "lease_renewal_date",
        "balance_owed",
    }
    assert spec.scenarios.call_reasons


def test_prospect_loads_and_validates() -> None:
    spec = load_category(PROSPECT_YAML)
    assert spec.category == "prospect"
    assert {f.name for f in spec.fields} == {
        "name",
        "phone",
        "email",
        "budget",
        "move_in_date",
        "bedroom_type",
        "pet_info",
        "parking_need",
        "tour_slot",
    }


def test_load_all_categories_discovers_all_files_in_directory() -> None:
    specs = load_all_categories()
    assert {s.category for s in specs} == {"resident", "prospect"}


def test_new_category_file_is_discovered_without_code_changes(tmp_path: Path) -> None:
    categories_dir = tmp_path / "categories"
    categories_dir.mkdir()
    (categories_dir / "resident.yaml").write_text(RESIDENT_YAML.read_text(encoding="utf-8"))

    vendor_yaml = {
        "category": "vendor",
        "fields": [
            {
                "name": "company_name",
                "match_type": "fuzzy",
                "required": True,
                "distractor_strategy": "swap_similar_name",
            }
        ],
        "scenarios": {"call_reasons": ["invoice_question"]},
    }
    (categories_dir / "vendor.yaml").write_text(yaml.safe_dump(vendor_yaml))

    specs = load_all_categories(categories_dir)
    assert {s.category for s in specs} == {"resident", "vendor"}


def test_missing_required_field_raises_schema_error(tmp_path: Path) -> None:
    bad_yaml = {
        "category": "broken",
        "fields": [
            {
                "name": "name",
                "match_type": "fuzzy",
                # missing "required" and "distractor_strategy"
            }
        ],
        "scenarios": {"call_reasons": ["some_reason"]},
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(bad_yaml))

    with pytest.raises(SchemaError):
        load_category(path)


def test_unknown_match_type_raises_schema_error(tmp_path: Path) -> None:
    bad_yaml = {
        "category": "broken",
        "fields": [
            {
                "name": "name",
                "match_type": "telepathic",
                "required": True,
                "distractor_strategy": "swap_similar_name",
            }
        ],
        "scenarios": {"call_reasons": ["some_reason"]},
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(bad_yaml))

    with pytest.raises(SchemaError):
        load_category(path)


def test_dangling_value_pool_reference_raises_schema_error(tmp_path: Path) -> None:
    bad_yaml = {
        "category": "broken",
        "fields": [
            {
                "name": "issue_type",
                "match_type": "canonical_list",
                "required": True,
                "distractor_strategy": "swap_sibling_value",
                "value_pool": "does_not_exist",
            }
        ],
        "scenarios": {"call_reasons": ["some_reason"]},
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(bad_yaml))

    with pytest.raises(SchemaError):
        load_category(path, value_pools_dir=VALUE_POOLS_DIR)


def test_canonical_list_field_without_value_pool_raises_schema_error(tmp_path: Path) -> None:
    bad_yaml = {
        "category": "broken",
        "fields": [
            {
                "name": "issue_type",
                "match_type": "canonical_list",
                "required": True,
                "distractor_strategy": "swap_sibling_value",
            }
        ],
        "scenarios": {"call_reasons": ["some_reason"]},
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(bad_yaml))

    with pytest.raises(SchemaError):
        load_category(path, value_pools_dir=VALUE_POOLS_DIR)
