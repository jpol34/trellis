"""Loads training-corpus JSONL records and translates each field into the dict-keyed
`ChoiceQuestion` wire shape via `trellis.reference.wire`, so fine-tuning never sees raw
positional `candidates`/`correct_index`. Deliberately free of torch/laya imports: this module
is exercised directly by fast unit tests, independent of the heavy training loop in `train.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trellis.generation.corpus_common import read_jsonl
from trellis.reference.wire import candidates_to_criteria

QUESTION_INSTRUCTIONS = "Which value applies for the {field!r} field?"


@dataclass(frozen=True)
class ChoiceExample:
    """One `choice`-question training example: a transcript (state), a wire-shaped criteria
    dict for one field's candidate set, and the criteria key holding the correct answer."""

    item_id: str
    field: str
    transcript: str
    instructions: str
    criteria: dict[str, str]
    correct_key: str


def record_to_examples(record: dict) -> list[ChoiceExample]:
    """Translates one training-corpus record's fields into one `ChoiceExample` per field,
    via the ticket #4 wire adapter (`candidates_to_criteria`)."""
    examples = []
    for field in record["fields"]:
        criteria = candidates_to_criteria(field["candidates"])
        correct_key = list(criteria.keys())[field["correct_index"]]
        examples.append(
            ChoiceExample(
                item_id=record["item_id"],
                field=field["field"],
                transcript=record["transcript"],
                instructions=QUESTION_INSTRUCTIONS.format(field=field["field"]),
                criteria=criteria,
                correct_key=correct_key,
            )
        )
    return examples


def load_examples(directory: Path) -> list[ChoiceExample]:
    """Loads every `*.jsonl` file directly under `directory` and translates all records into
    `ChoiceExample`s, in file-then-record-then-field order (deterministic given a fixed
    directory listing, so slicing a fixed prefix for a dry run is reproducible)."""
    examples: list[ChoiceExample] = []
    for path in sorted(directory.glob("*.jsonl")):
        for record in read_jsonl(path):
            examples.extend(record_to_examples(record))
    return examples
