"""CLI entry point (`trellis-eval`) for #15: the end-to-end three-arm benchmark run.

Loads the held-out eval set (#7's `build_eval_set.py` output), then for every (transcript,
field) pair dispatches every arm currently registered in `reference.base.ARMS` (today
`trellis`/`gpt-5.1`/`jev`, generically extensible) through the scoring path its declared `mode`
selects: `closed_set.py` for closed-set arms (given the field's candidate list) or `matchers.py`
for open-extraction arms (asked to extract free text with no candidates). Every prediction
resolves to one of `states.FiveState`'s 5 outcomes, `metrics.py` aggregates those into
accuracy/precision/recall/F1, and `report.py` renders the three-arm markdown comparison.

Result dataclass shape (`ItemResult`/`DatasetResult`/`BackendResult`) mirrors laya-bench's
`eval/runner.py`, with the "backend" axis being whichever arm names are currently registered and
the "dataset" axis being the eval set's categories (resident/prospect).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from trellis.generation.corpus_common import read_jsonl
from trellis.generation.distractors import NOT_MENTIONED
from trellis.generation.values import load_value_pool

# Importing these arm modules is what triggers their `register_arm()` call into `ARMS` (see
# `reference/base.py`) — nothing else in this file references them by name, so without these
# imports `ARMS` would be empty for any process that only ever imports `run_eval`.
from trellis.reference import gpt51_arm, jev_arm, trellis_arm  # noqa: F401
from trellis.reference.base import ARMS, ReferenceArm
from trellis.schema.loader import load_all_categories
from trellis.schema.types import FieldSpec
from trellis.validate import matchers
from trellis.validate.closed_set import score_closed_set
from trellis.validate.report import render_markdown
from trellis.validate.states import FiveState, resolve_state

DEFAULT_EVAL_DIR = Path("data") / "eval_set"

# `numeric_tolerance_match`'s tolerance is a scoring-time parameter, not part of `FieldSpec` —
# generation's own perturbation magnitude (see `distractors.py::_perturb_amount`) is randomized
# and strategy-internal, not a fixed value scoring can reuse. $0.50 matches the tolerance already
# established by `tests/unit/test_matchers.py`'s numeric_tolerance_match examples.
NUMERIC_TOLERANCE = 0.5


def _match(field: FieldSpec, predicted: str, gold: str) -> bool:
    if field.match_type == "exact":
        return matchers.exact_match(predicted, gold)
    if field.match_type == "normalized_phone":
        return matchers.normalized_phone_match(predicted, gold)
    if field.match_type == "fuzzy":
        return matchers.fuzzy_match(predicted, gold)
    if field.match_type == "canonical_list":
        assert field.value_pool is not None
        return matchers.canonical_list_match(predicted, gold, load_value_pool(field.value_pool))
    if field.match_type == "numeric_tolerance":
        return matchers.numeric_tolerance_match(predicted, gold, tolerance=NUMERIC_TOLERANCE)
    if field.match_type == "date":
        return matchers.date_match(predicted, gold)
    raise ValueError(f"no matcher for match_type {field.match_type!r}")  # pragma: no cover


@dataclass
class ItemResult:
    item_id: str
    category: str
    field: str
    predicted_value: str | None
    gold_value: str
    confidence: float | None
    latency_ms: float | None
    state: FiveState | None  # None when `error` is set: the arm call raised before scoring
    error: str | None = None


@dataclass
class DatasetResult:
    dataset_name: str  # eval-set category (resident/prospect)
    items: list[ItemResult] = dc_field(default_factory=list)


@dataclass
class BackendResult:
    backend_name: str  # arm name
    mode: str
    datasets: dict[str, DatasetResult] = dc_field(default_factory=dict)


def _field_specs_by_category(
    categories: list,
) -> dict[str, dict[str, FieldSpec]]:
    return {c.category: {f.name: f for f in c.fields} for c in categories}


async def _score_one(
    arm: ReferenceArm,
    record: dict,
    fld: dict,
    field_spec: FieldSpec,
) -> ItemResult:
    item_id = record["item_id"]
    category = record["category"]
    field_name = fld["field"]
    candidates: list[str] = fld["candidates"]
    gold_value: str = fld["value"]

    started = time.perf_counter()
    try:
        if arm.mode == "closed_set":
            answer = await arm.answer(record["transcript"], field_spec, candidates)
        elif arm.mode == "open_extraction":
            answer = await arm.answer(record["transcript"], field_spec, None)
        else:
            raise ValueError(f"arm {arm.name!r} has unrecognized mode {arm.mode!r}")
    except Exception as e:  # noqa: BLE001 - a single arm/item failure must not abort the run
        latency_ms = (time.perf_counter() - started) * 1000
        return ItemResult(
            item_id=item_id,
            category=category,
            field=field_name,
            predicted_value=None,
            gold_value=gold_value,
            confidence=None,
            latency_ms=latency_ms,
            state=None,
            error=f"{type(e).__name__}: {e}",
        )
    latency_ms = (time.perf_counter() - started) * 1000

    field_required_and_stated_in_gold = gold_value != NOT_MENTIONED

    if arm.mode == "closed_set":
        not_mentioned_index = (
            candidates.index(NOT_MENTIONED) if NOT_MENTIONED in candidates else None
        )
        state = score_closed_set(
            chosen_index=answer["chosen_index"],
            correct_index=fld["correct_index"],
            not_mentioned_index=not_mentioned_index,
        )
    else:
        model_abstained = answer["value"] is None
        if model_abstained:
            model_correct = None
        else:
            model_correct = _match(field_spec, answer["value"], gold_value)
        state = resolve_state(field_required_and_stated_in_gold, model_abstained, model_correct)

    return ItemResult(
        item_id=item_id,
        category=category,
        field=field_name,
        predicted_value=answer["value"],
        gold_value=gold_value,
        confidence=answer["confidence"],
        latency_ms=latency_ms,
        state=state,
    )


async def run_arm(
    arm: ReferenceArm,
    records: list[dict],
    field_specs: dict[str, dict[str, FieldSpec]],
    concurrency: int,
) -> BackendResult:
    result = BackendResult(backend_name=arm.name, mode=arm.mode)
    tasks: list[tuple[str, asyncio.Task[ItemResult]]] = []
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(record: dict, fld: dict, field_spec: FieldSpec) -> ItemResult:
        async with semaphore:
            return await _score_one(arm, record, fld, field_spec)

    async with asyncio.TaskGroup() as tg:
        for record in records:
            category = record["category"]
            for fld in record["fields"]:
                field_spec = field_specs[category][fld["field"]]
                task = tg.create_task(_bounded(record, fld, field_spec))
                tasks.append((category, task))

    for category, task in tasks:
        result.datasets.setdefault(category, DatasetResult(dataset_name=category)).items.append(
            task.result()
        )
    return result


async def run_all(
    records: list[dict],
    field_specs: dict[str, dict[str, FieldSpec]],
    arms: dict[str, ReferenceArm] | None = None,
    concurrency: int = 5,
) -> dict[str, BackendResult]:
    arms = ARMS if arms is None else arms
    results: dict[str, BackendResult] = {}
    for name, arm in arms.items():
        results[name] = await run_arm(arm, records, field_specs, concurrency)
    return results


def load_eval_set(eval_dir: Path) -> list[dict]:
    records: list[dict] = []
    paths = sorted(eval_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl eval-set files found under {eval_dir}")
    for path in paths:
        records.extend(read_jsonl(path))
    return records


async def run_eval(
    eval_dir: Path = DEFAULT_EVAL_DIR,
    arms: dict[str, ReferenceArm] | None = None,
    concurrency: int = 5,
    limit: int | None = None,
) -> tuple[dict[str, BackendResult], list[str]]:
    records = load_eval_set(eval_dir)
    if limit is not None:
        records = records[:limit]
    field_specs = _field_specs_by_category(load_all_categories())
    categories = sorted({r["category"] for r in records})
    results = await run_all(records, field_specs, arms=arms, concurrency=concurrency)
    return results, categories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every registered reference arm (trellis/gpt-5.1/jev) against the held-"
        "out eval set and render a three-arm markdown comparison report."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--arm",
        action="append",
        dest="arm_names",
        default=None,
        help="Restrict the run to this arm name (repeatable). Default: every registered arm.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent answer() calls in flight per arm.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Score only the first N eval-set records, for a quick smoke run.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the report here instead of stdout."
    )
    args = parser.parse_args()

    if args.arm_names:
        missing = [n for n in args.arm_names if n not in ARMS]
        if missing:
            raise SystemExit(f"unknown arm(s) {missing!r}; registered arms: {sorted(ARMS)}")
        arms = {n: ARMS[n] for n in args.arm_names}
    else:
        arms = None

    results, categories = asyncio.run(
        run_eval(
            eval_dir=args.eval_dir, arms=arms, concurrency=args.concurrency, limit=args.limit
        )
    )
    report = render_markdown(results, categories)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
