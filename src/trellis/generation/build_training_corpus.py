"""CLI entry point (`trellis-gen-training`) driving #5/#6's generation pipeline into the full
training corpus: 400 transcripts/category (residents + prospects = 800 total), evenly sampled
across call_reason x variability_tier scenario cells, run through the quality gates, then split
80/10/10 into train/val/held_out_internal independently per scenario cell with a fixed seed.

Must never import from `build_eval_set.py` — the two generate and split independently, sharing
only `corpus_common`, `quality_gates`, and `transcript_gen` (see
`tests/unit/test_import_isolation.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from trellis.generation.corpus_common import (
    GenerateFn,
    allocate_cells_flat,
    generate_records,
    record_to_quality_gate_item,
    stratified_split,
    write_jsonl,
)
from trellis.generation.quality_gates import run_quality_gates
from trellis.generation.transcript_gen import generate_scenario_item
from trellis.schema.loader import load_all_categories

TRANSCRIPTS_PER_CATEGORY = 400
# Fixed so which cells get an allocation remainder, and split membership, are reproducible.
SAMPLE_SEED = 700070
SPLIT_SEED = 700071

DEFAULT_OUT_DIR = Path("data") / "training_corpus"


async def build_training_corpus(
    *,
    total: int = TRANSCRIPTS_PER_CATEGORY * 2,
    out_dir: Path = DEFAULT_OUT_DIR,
    generate_fn: GenerateFn = generate_scenario_item,
    sample_seed: int = SAMPLE_SEED,
    split_seed: int = SPLIT_SEED,
) -> dict[str, int]:
    categories = load_all_categories()
    records = await generate_records(
        categories,
        total,
        generate_fn=generate_fn,
        rng=random.Random(sample_seed),
        id_prefix="train",
        allocate_cells=allocate_cells_flat,
    )

    run_quality_gates(
        [record_to_quality_gate_item(r) for r in records],
        diversity_review_path=out_dir / "diversity_review.json",
        injection_review_path=out_dir / "injection_review.json",
        distractor_review_path=out_dir / "distractor_review.json",
    )

    train, val, held_out = stratified_split(records, split_seed)
    return {
        "train": write_jsonl(out_dir / "train" / "records.jsonl", train),
        "val": write_jsonl(out_dir / "val" / "records.jsonl", val),
        "held_out_internal": write_jsonl(
            out_dir / "held_out_internal" / "records.jsonl", held_out
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the training corpus and split it 80/10/10 per scenario cell "
        "into train/val/held_out_internal."
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        metavar="N",
        help="Generate only N transcripts total instead of the full "
        f"{TRANSCRIPTS_PER_CATEGORY * 2}, for a manually-reviewed pilot batch.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    total = args.pilot if args.pilot is not None else TRANSCRIPTS_PER_CATEGORY * 2
    counts = asyncio.run(build_training_corpus(total=total, out_dir=args.out_dir))
    print(
        f"train: {counts['train']}  val: {counts['val']}  "
        f"held_out_internal: {counts['held_out_internal']}"
    )


if __name__ == "__main__":
    main()
