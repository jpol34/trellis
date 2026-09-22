"""Shared building blocks for `build_training_corpus.py` and `build_eval_set.py`: even
allocation of a target item count across scenario cells (call_reason x variability_tier), a
seeded stratified train/val/held-out split, and JSONL record I/O. Both callers import from here
but never from each other — see `tests/unit/test_import_isolation.py`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
from collections.abc import Awaitable, Callable, Hashable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from trellis.generation.quality_gates import QualityGateField, QualityGateItem
from trellis.schema.types import CategorySpec

VARIABILITY_TIERS: tuple[str, str, str] = ("clean", "natural", "noisy")

# (category, call_reason, variability_tier) -> generated item, as returned by
# `trellis.generation.transcript_gen.generate_scenario_item` (or a stand-in with the same
# signature, for tests).
GenerateFn = Callable[[CategorySpec, str, str], Awaitable[dict]]

_Cell = TypeVar("_Cell", bound=Hashable)


def allocate_counts(cells: Sequence[_Cell], total: int, rng: random.Random) -> dict[_Cell, int]:
    """Splits `total` as evenly as integer counts allow across `cells`. Any remainder is handed
    out one-per-cell to a seeded-random subset of `cells`, so the +1s are reproducible given
    `rng` without always landing on the same cells by iteration order."""
    if not cells:
        if total:
            raise ValueError("no cells to allocate a nonzero total across")
        return {}
    base, remainder = divmod(total, len(cells))
    order = list(cells)
    rng.shuffle(order)
    bumped = set(order[:remainder])
    return {cell: base + (1 if cell in bumped else 0) for cell in cells}


def cells_flat(category: CategorySpec) -> list[tuple[str, str]]:
    return [(cr, tier) for cr in category.scenarios.call_reasons for tier in VARIABILITY_TIERS]


def allocate_cells_flat(
    category: CategorySpec, total: int, rng: random.Random
) -> dict[tuple[str, str], int]:
    """Distributes `total` across every (call_reason, variability_tier) cell for `category` in
    one flat pass."""
    return allocate_counts(cells_flat(category), total, rng)


def allocate_cells_by_tier(
    category: CategorySpec, total: int, rng: random.Random
) -> dict[tuple[str, str], int]:
    """Splits `total` across variability tiers first, then across call reasons within each
    tier — guarantees an even per-tier split (e.g. 50 -> 17/17/16) rather than leaving tier
    totals to fall out incidentally from a flat cell allocation."""
    tier_counts = allocate_counts(list(VARIABILITY_TIERS), total, rng)
    result: dict[tuple[str, str], int] = {}
    for tier, tier_total in tier_counts.items():
        reason_counts = allocate_counts(category.scenarios.call_reasons, tier_total, rng)
        for call_reason, count in reason_counts.items():
            result[(call_reason, tier)] = count
    return result


AllocateCellsFn = Callable[[CategorySpec, int, random.Random], dict[tuple[str, str], int]]


def _persona_to_dict(persona: object) -> dict:
    return dataclasses.asdict(persona) if dataclasses.is_dataclass(persona) else dict(persona)


def build_record(
    *,
    item_id: str,
    category: CategorySpec,
    call_reason: str,
    variability_tier: str,
    generated: dict,
) -> dict:
    """Shapes one `generate_fn` result into the flat JSONL record schema written to disk."""
    distractor_strategies = {f.name: f.distractor_strategy for f in category.fields}
    fields = [
        {
            "field": gt["field"],
            "value": gt["value"],
            "candidates": gt["candidates"],
            "correct_index": gt["correct_index"],
            "char_offset": gt["char_offset"],
            "distractor_strategy": distractor_strategies.get(gt["field"], ""),
        }
        for gt in generated["ground_truth"]
    ]
    return {
        "item_id": item_id,
        "category": category.category,
        "call_reason": call_reason,
        "variability_tier": variability_tier,
        "transcript": generated["transcript"],
        "persona": _persona_to_dict(generated["persona"]),
        "fields": fields,
    }


def record_to_quality_gate_item(record: dict) -> QualityGateItem:
    return QualityGateItem(
        item_id=record["item_id"],
        category=record["category"],
        call_reason=record["call_reason"],
        variability_tier=record["variability_tier"],
        transcript=record["transcript"],
        fields=[
            QualityGateField(
                field=f["field"],
                value=f["value"],
                candidates=f["candidates"],
                char_offset=f["char_offset"],
                distractor_strategy=f.get("distractor_strategy", ""),
            )
            for f in record["fields"]
        ],
    )


def _load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    if not checkpoint_path.exists():
        return {}
    records: dict[str, dict] = {}
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            item_id = record["item_id"]
        except (json.JSONDecodeError, KeyError):
            # A process killed mid-write (SIGKILL, OOM, power loss) can leave a truncated or
            # otherwise malformed last line. Treat it as never-committed rather than failing the
            # whole resume — the item it would have been simply gets regenerated.
            continue
        records[item_id] = record
    return records


async def generate_records(
    categories: Sequence[CategorySpec],
    total: int,
    *,
    generate_fn: GenerateFn,
    rng: random.Random,
    id_prefix: str,
    allocate_cells: AllocateCellsFn = allocate_cells_flat,
    checkpoint_path: Path | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Allocates `total` across `categories` (evenly, seeded), then across each category's
    scenario cells via `allocate_cells`, calling `generate_fn` once per generated item.

    A real generation run is many billed API calls; without `checkpoint_path`, any single
    failure partway through (a parse error, a network blip, exhausted retries) loses every item
    generated so far, since nothing is persisted until the whole run finishes. When given, each
    successfully generated record is appended to `checkpoint_path` immediately, and a checkpoint
    already on disk from a prior interrupted run is loaded first so already-generated items are
    skipped rather than regenerated (and re-billed). Only checkpoint entries whose item_id is
    actually part of *this* call's allocation are reused — a stale checkpoint left over from a
    run with a different `total` (e.g. a `--pilot N` run sharing the same out_dir as a later
    full run) must not inflate the result past `total` or leak mismatched items in.

    Pending (not-yet-checkpointed) items run under `concurrency`-bounded fan-out: the first
    pending item runs alone first (preflight — fails fast on a bad key/config instead of firing
    `concurrency` calls that all hit the same error), then the rest run concurrently through an
    `asyncio.TaskGroup`, each bounded by an `asyncio.Semaphore(concurrency)`. A `TaskGroup` is
    used instead of `asyncio.gather` so that the first failure cancels every other in-flight
    task rather than leaving them running after the caller has already seen the exception; the
    resulting `ExceptionGroup` is unwrapped so callers see the same exception type a sequential
    run would raise. `concurrency=1` reproduces the original strictly-sequential call order and
    checkpoint contents exactly. A single `asyncio.Lock()` guards every checkpoint append+flush
    for the duration of the call — safe under concurrent completions since checkpoint resume
    already keys strictly by `item_id`, not file position/line order."""
    on_disk_checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    per_category_total = allocate_counts([c.category for c in categories], total, rng)

    plan: list[tuple[str, CategorySpec, str, str]] = []
    for category in categories:
        cell_counts = allocate_cells(category, per_category_total[category.category], rng)
        for (call_reason, tier), count in cell_counts.items():
            for i in range(count):
                item_id = f"{id_prefix}-{category.category}-{call_reason}-{tier}-{i}"
                plan.append((item_id, category, call_reason, tier))

    records_by_id: dict[str, dict] = {}
    pending: list[tuple[str, CategorySpec, str, str]] = []
    for item_id, category, call_reason, tier in plan:
        if item_id in on_disk_checkpoint:
            records_by_id[item_id] = on_disk_checkpoint[item_id]
        else:
            pending.append((item_id, category, call_reason, tier))

    checkpoint_file = None
    checkpoint_lock = asyncio.Lock()
    try:
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_path.open("a", encoding="utf-8")

        async def run_one(
            item_id: str, category: CategorySpec, call_reason: str, tier: str
        ) -> None:
            generated = await generate_fn(category, call_reason, tier)
            record = build_record(
                item_id=item_id,
                category=category,
                call_reason=call_reason,
                variability_tier=tier,
                generated=generated,
            )
            if checkpoint_file:
                async with checkpoint_lock:
                    checkpoint_file.write(json.dumps(record) + "\n")
                    checkpoint_file.flush()
            records_by_id[item_id] = record

        if pending:
            first, *rest = pending
            await run_one(*first)

            if rest and concurrency <= 1:
                # A semaphore alone doesn't guarantee serialization here: if a task never
                # actually suspends (e.g. a fully synchronous generate_fn), sibling cancellation
                # from TaskGroup is delivered a full event-loop tick too late to stop it from
                # already having run to completion. Running the plain sequential loop instead
                # guarantees `concurrency=1` is bit-for-bit today's behavior, not just "usually
                # equivalent."
                for args in rest:
                    await run_one(*args)
            elif rest:
                semaphore = asyncio.Semaphore(concurrency)

                async def run_bounded(args: tuple[str, CategorySpec, str, str]) -> None:
                    async with semaphore:
                        await run_one(*args)

                try:
                    async with asyncio.TaskGroup() as tg:
                        for args in rest:
                            tg.create_task(run_bounded(args))
                except ExceptionGroup as eg:
                    raise eg.exceptions[0] from None
    finally:
        if checkpoint_file:
            checkpoint_file.close()
    return [records_by_id[item_id] for item_id, *_ in plan]


def default_cell_key(record: dict) -> tuple[str, str, str]:
    return (record["category"], record["call_reason"], record["variability_tier"])


def stratified_split(
    records: list[dict],
    seed: int,
    *,
    cell_key: Callable[[dict], Hashable] = default_cell_key,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[dict], list[dict], list[dict]]:
    """Splits `records` into three lists per `ratios`, independently within each `cell_key`
    group (rather than over the whole pool) so a cell can't end up entirely absent from one
    split by chance. `seed` fixes both the per-cell shuffle and the split, so the result is
    reproducible."""
    rng = random.Random(seed)
    by_cell: dict[Hashable, list[dict]] = {}
    for record in records:
        by_cell.setdefault(cell_key(record), []).append(record)

    train: list[dict] = []
    val: list[dict] = []
    held_out: list[dict] = []
    for cell in sorted(by_cell, key=repr):
        items = by_cell[cell][:]
        rng.shuffle(items)
        n = len(items)
        n_val = round(n * ratios[1])
        n_held = round(n * ratios[2])
        n_train = n - n_val - n_held
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        held_out.extend(items[n_train + n_val :])
    return train, val, held_out


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
            count += 1
    return count
