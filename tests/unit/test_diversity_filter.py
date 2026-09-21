from __future__ import annotations

import json

from trellis.generation.diversity_filter import (
    ScenarioItem,
    cosine_distance,
    find_near_duplicates,
    tfidf_embedder,
    write_diversity_review,
)

CELL = dict(category="prospect", call_reason="new_inquiry", variability_tier="clean")

NEAR_DUP_A = (
    "Caller: Hi, I'm calling about the two bedroom unit. My move-in date is flexible, "
    "probably sometime next month. Agent: Sure, I can help with that."
)
NEAR_DUP_B = (
    "Caller: Hi, I'm calling about the two bedroom unit. My move-in date is flexible, "
    "probably sometime next month, maybe a bit later. Agent: Sure, I can help with that."
)
DISTINCT = (
    "Caller: Yeah so, uh, my dishwasher's been leaking for like a week now and nobody's "
    "called me back, this is getting pretty frustrating honestly. Agent: I'm sorry to hear "
    "that, let's get a work order started right away."
)


def test_cosine_distance_identical_vectors_is_zero():
    assert cosine_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_distance_orthogonal_vectors_is_one():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_cosine_distance_zero_vector_is_maximally_distant():
    assert cosine_distance([0.0, 0.0], [1.0, 2.0]) == 1.0


def test_near_duplicate_pair_flagged_within_cell():
    items = [
        ScenarioItem(item_id="a", transcript=NEAR_DUP_A, **CELL),
        ScenarioItem(item_id="b", transcript=NEAR_DUP_B, **CELL),
    ]
    pairs = find_near_duplicates(items, threshold=0.15)
    assert len(pairs) == 1
    assert {pairs[0].item_a, pairs[0].item_b} == {"a", "b"}


def test_distinct_pair_not_flagged():
    items = [
        ScenarioItem(item_id="a", transcript=NEAR_DUP_A, **CELL),
        ScenarioItem(item_id="c", transcript=DISTINCT, **CELL),
    ]
    pairs = find_near_duplicates(items, threshold=0.15)
    assert pairs == []


def test_cross_cell_similarity_never_compared():
    other_cell = dict(category="prospect", call_reason="new_inquiry", variability_tier="noisy")
    items = [
        ScenarioItem(item_id="a", transcript=NEAR_DUP_A, **CELL),
        ScenarioItem(item_id="b", transcript=NEAR_DUP_A, **other_cell),
    ]
    assert find_near_duplicates(items, threshold=0.15) == []


def test_single_item_cell_produces_no_pairs():
    items = [ScenarioItem(item_id="a", transcript=NEAR_DUP_A, **CELL)]
    assert find_near_duplicates(items) == []


def test_tfidf_embedder_produces_one_vector_per_text_same_dimension():
    vectors = tfidf_embedder([NEAR_DUP_A, DISTINCT])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])


def test_write_diversity_review_preserves_reviewed_entry_and_adds_new_flags(tmp_path):
    path = tmp_path / "diversity_review.json"
    items = [
        ScenarioItem(item_id="a", transcript=NEAR_DUP_A, **CELL),
        ScenarioItem(item_id="b", transcript=NEAR_DUP_B, **CELL),
    ]
    pairs = find_near_duplicates(items, threshold=0.15)
    write_diversity_review(pairs, path)

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["reviewed"] is None

    entries[0]["reviewed"] = True
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    write_diversity_review(pairs, path)
    entries_after = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries_after) == 1
    assert entries_after[0]["reviewed"] is True

    other_cell = dict(category="prospect", call_reason="new_inquiry", variability_tier="noisy")
    items_with_new = [
        ScenarioItem(item_id="a", transcript=NEAR_DUP_A, **CELL),
        ScenarioItem(item_id="b", transcript=NEAR_DUP_B, **CELL),
        ScenarioItem(item_id="c", transcript=NEAR_DUP_A, **other_cell),
        ScenarioItem(item_id="d", transcript=NEAR_DUP_B, **other_cell),
    ]
    write_diversity_review(find_near_duplicates(items_with_new, threshold=0.15), path)
    entries_final = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries_final) == 2
    reviewed_by_pair = {(e["item_a"], e["item_b"]): e["reviewed"] for e in entries_final}
    assert reviewed_by_pair[("a", "b")] is True
    assert reviewed_by_pair[("c", "d")] is None
