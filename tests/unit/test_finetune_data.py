from __future__ import annotations

import json

from trellis.finetune.data import QUESTION_INSTRUCTIONS, load_examples, record_to_examples
from trellis.generation.corpus_common import read_jsonl, write_jsonl
from trellis.reference.wire import candidates_to_criteria


def _record(item_id: str, fields: list[dict]) -> dict:
    return {
        "item_id": item_id,
        "category": "resident",
        "call_reason": "maintenance",
        "variability_tier": "clean",
        "transcript": f"transcript for {item_id}",
        "persona": {"age_range": "30-40", "tone": "calm"},
        "fields": fields,
    }


def _field(name: str, candidates: list[str], correct_index: int) -> dict:
    return {
        "field": name,
        "value": candidates[correct_index],
        "candidates": candidates,
        "correct_index": correct_index,
        "char_offset": 0,
        "distractor_strategy": "swap_sibling_value",
    }


def test_read_jsonl_round_trips_write_jsonl(tmp_path):
    records = [_record("a", [_field("issue_type", ["leak", "noise"], 0)])]
    path = tmp_path / "records.jsonl"
    count = write_jsonl(path, records)

    assert count == 1
    assert read_jsonl(path) == records


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_record_to_examples_translates_via_wire_adapter():
    candidates = ["leak", "noise", "not mentioned"]
    record = _record("item-1", [_field("issue_type", candidates, 1)])

    examples = record_to_examples(record)

    assert len(examples) == 1
    ex = examples[0]
    assert ex.item_id == "item-1"
    assert ex.field == "issue_type"
    assert ex.transcript == "transcript for item-1"
    assert ex.instructions == QUESTION_INSTRUCTIONS.format(field="issue_type")
    # Must go through the ticket #4 wire adapter, not raw positional candidates.
    assert ex.criteria == candidates_to_criteria(candidates)
    assert ex.correct_key == "candidate_1"


def test_record_to_examples_produces_one_example_per_field():
    fields = [
        _field("issue_type", ["leak", "noise"], 0),
        _field("severity", ["low", "medium", "high"], 2),
    ]
    record = _record("item-2", fields)

    examples = record_to_examples(record)

    assert [ex.field for ex in examples] == ["issue_type", "severity"]
    assert examples[1].correct_key == "candidate_2"
    assert examples[1].criteria == candidates_to_criteria(["low", "medium", "high"])


def test_load_examples_reads_every_jsonl_file_in_directory(tmp_path):
    write_jsonl(
        tmp_path / "a.jsonl",
        [_record("a1", [_field("f", ["x", "y"], 0)])],
    )
    write_jsonl(
        tmp_path / "b.jsonl",
        [_record("b1", [_field("f", ["x", "y", "z"], 2)])],
    )

    examples = load_examples(tmp_path)

    assert {ex.item_id for ex in examples} == {"a1", "b1"}


def test_load_examples_empty_directory_returns_empty_list(tmp_path):
    assert load_examples(tmp_path) == []


def test_record_to_examples_matches_json_round_trip(tmp_path):
    """Records loaded back from disk (str keys, plain dicts) translate the same as
    in-memory records - guards against any float/int/JSON-shape drift through the wire adapter."""
    candidates = ["a", "b", "c", "d", "not mentioned"]
    record = _record("item-3", [_field("field", candidates, 3)])
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [record])

    (on_disk,) = read_jsonl(path)
    examples = record_to_examples(on_disk)

    assert examples[0].correct_key == "candidate_3"
    assert examples[0].criteria == json.loads(json.dumps(candidates_to_criteria(candidates)))
