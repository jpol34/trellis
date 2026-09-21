"""Batch-quality layer over #5's transcript generator (`trellis.generation.transcript_gen`):
runs the diversity filter and both leakage-check heuristics over a batch of already-generated
items and writes their reports as human-editable review files.

CLI-callable via `trellis-quality-gates <batch.json> [--out-dir DIR]`, where `batch.json` is a
JSON list of items shaped like `QualityGateItem` (see `_load_batch`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from trellis.generation.diversity_filter import (
    Embedder,
    NearDuplicatePair,
    ScenarioItem,
    find_near_duplicates,
    tfidf_embedder,
    write_diversity_review,
)
from trellis.generation.leakage_check import (
    DistractorTellFlag,
    InjectionCleanlinessFlag,
    LeakageCheckField,
    LeakageCheckItem,
    run_leakage_check,
    write_distractor_review,
    write_injection_review,
)


@dataclass(frozen=True)
class QualityGateField:
    field: str
    value: str
    candidates: list[str]
    char_offset: int | None


@dataclass(frozen=True)
class QualityGateItem:
    item_id: str
    category: str
    call_reason: str
    variability_tier: str
    transcript: str
    fields: list[QualityGateField] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class QualityGateReport:
    near_duplicates: list[NearDuplicatePair]
    injection_flags: list[InjectionCleanlinessFlag]
    distractor_flags: list[DistractorTellFlag]


def run_quality_gates(
    items: list[QualityGateItem],
    *,
    diversity_review_path: Path,
    injection_review_path: Path,
    distractor_review_path: Path,
    embedder: Embedder = tfidf_embedder,
    diversity_threshold: float = 0.15,
) -> QualityGateReport:
    """Runs the diversity filter and both leakage-check heuristics over `items` and writes all
    three human-editable review files, preserving already-reviewed entries from prior runs at
    the same paths."""
    scenario_items = [
        ScenarioItem(
            item_id=i.item_id,
            transcript=i.transcript,
            category=i.category,
            call_reason=i.call_reason,
            variability_tier=i.variability_tier,
        )
        for i in items
    ]
    near_duplicates = find_near_duplicates(
        scenario_items, embedder=embedder, threshold=diversity_threshold
    )
    write_diversity_review(near_duplicates, diversity_review_path)

    leakage_items = [
        LeakageCheckItem(
            item_id=i.item_id,
            transcript=i.transcript,
            variability_tier=i.variability_tier,
            fields=[
                LeakageCheckField(
                    field=f.field,
                    value=f.value,
                    candidates=f.candidates,
                    char_offset=f.char_offset,
                )
                for f in i.fields
            ],
        )
        for i in items
    ]
    injection_flags, distractor_flags = run_leakage_check(leakage_items)
    write_injection_review(injection_flags, injection_review_path)
    write_distractor_review(distractor_flags, distractor_review_path)

    return QualityGateReport(
        near_duplicates=near_duplicates,
        injection_flags=injection_flags,
        distractor_flags=distractor_flags,
    )


def _load_batch(path: Path) -> list[QualityGateItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        QualityGateItem(
            item_id=entry["item_id"],
            category=entry["category"],
            call_reason=entry["call_reason"],
            variability_tier=entry["variability_tier"],
            transcript=entry["transcript"],
            fields=[
                QualityGateField(
                    field=f["field"],
                    value=f["value"],
                    candidates=f["candidates"],
                    char_offset=f.get("char_offset"),
                )
                for f in entry["fields"]
            ],
        )
        for entry in raw
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diversity + leakage quality gates over a batch of generated items."
    )
    parser.add_argument(
        "batch",
        type=Path,
        help="JSON file: a list of generated items (item_id, category, call_reason, "
        "variability_tier, transcript, fields).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("quality_gate_reports"))
    args = parser.parse_args()

    items = _load_batch(args.batch)
    report = run_quality_gates(
        items,
        diversity_review_path=args.out_dir / "diversity_review.json",
        injection_review_path=args.out_dir / "injection_review.json",
        distractor_review_path=args.out_dir / "distractor_review.json",
    )
    print(
        f"near-duplicate pairs flagged: {len(report.near_duplicates)}\n"
        f"injection-cleanliness flags: {len(report.injection_flags)}\n"
        f"distractor-tell flags: {len(report.distractor_flags)}"
    )


if __name__ == "__main__":
    main()
