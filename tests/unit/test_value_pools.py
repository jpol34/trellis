from __future__ import annotations

import yaml

from trellis.schema.loader import CATEGORIES_DIR, VALUE_POOLS_DIR, load_all_categories

REPO_ROOT = VALUE_POOLS_DIR.parents[1]
VARIABILITY_TIERS_YAML = REPO_ROOT / "configs" / "variability_tiers.yaml"


def test_every_canonical_list_value_pool_reference_resolves_to_a_nonempty_file() -> None:
    specs = load_all_categories(CATEGORIES_DIR)
    for spec in specs:
        for field in spec.fields:
            if field.match_type != "canonical_list":
                continue
            pool_path = VALUE_POOLS_DIR / f"{field.value_pool}.yaml"
            assert pool_path.is_file()
            values = yaml.safe_load(pool_path.read_text(encoding="utf-8"))
            assert values, f"{pool_path} is empty"


def test_variability_tiers_are_distinct_and_composable() -> None:
    tiers = yaml.safe_load(VARIABILITY_TIERS_YAML.read_text(encoding="utf-8"))
    assert set(tiers) == {"clean", "natural", "noisy"}

    tag_sets = {}
    for name, tags in tiers.items():
        assert set(tags) == {"linguistic", "conversational"}
        tag_sets[name] = frozenset(tags["linguistic"]) | frozenset(tags["conversational"])

    assert tag_sets["clean"] < tag_sets["natural"] < tag_sets["noisy"]
    assert len({tag_sets["clean"], tag_sets["natural"], tag_sets["noisy"]}) == 3
