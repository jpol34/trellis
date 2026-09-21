"""Shared human-review-file plumbing for quality-gate flags.

A flag is a JSON object with a stable `flag_id` plus arbitrary detail fields and a `reviewed`
field. A freshly-flagged item has `reviewed: null`; a human fills it in with `true`/`false` (or
any non-null value meaning "looked at this"). Re-running the check must not clobber an
already-reviewed entry, and must drop entries that are no longer flagged (resolved on this run
or the item dropped out of the batch) — mirrors laya-bench's
`write_disagreements_for_review`/`merge_reviewed_labels` pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reviewed(path: Path) -> dict[str, Any]:
    """Reads a review file into a flag_id -> reviewed map. Returns {} if the file doesn't exist
    yet (first run)."""
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["flag_id"]: entry["reviewed"] for entry in entries}


def write_flagged(flags: list[dict[str, Any]], path: Path) -> None:
    """Writes the current batch's flags to `path`, carrying forward any existing `reviewed`
    value for a flag_id already present in the file. Each dict in `flags` must have a
    `flag_id` key; every other key is written through unchanged."""
    existing = load_reviewed(path)
    out = []
    for flag in flags:
        entry = dict(flag)
        entry["reviewed"] = existing.get(flag["flag_id"])
        out.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
