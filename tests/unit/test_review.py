from __future__ import annotations

import json

from trellis.generation.review import load_reviewed, write_flagged


def test_load_reviewed_missing_file_returns_empty_dict(tmp_path):
    assert load_reviewed(tmp_path / "nope.json") == {}


def test_write_flagged_new_entries_have_null_reviewed(tmp_path):
    path = tmp_path / "review.json"
    write_flagged([{"flag_id": "x", "detail": "one"}], path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert entries == [{"flag_id": "x", "detail": "one", "reviewed": None}]


def test_write_flagged_preserves_reviewed_value_across_reruns(tmp_path):
    path = tmp_path / "review.json"
    write_flagged([{"flag_id": "x", "detail": "one"}, {"flag_id": "y", "detail": "two"}], path)

    entries = json.loads(path.read_text(encoding="utf-8"))
    for e in entries:
        if e["flag_id"] == "x":
            e["reviewed"] = True
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    write_flagged([{"flag_id": "x", "detail": "one"}, {"flag_id": "y", "detail": "two"}], path)
    reviewed = load_reviewed(path)
    assert reviewed == {"x": True, "y": None}


def test_write_flagged_drops_entries_no_longer_flagged(tmp_path):
    path = tmp_path / "review.json"
    write_flagged([{"flag_id": "x", "detail": "one"}], path)
    write_flagged([{"flag_id": "y", "detail": "two"}], path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert [e["flag_id"] for e in entries] == ["y"]
