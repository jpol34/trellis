"""CLI entry point (`trellis-gen-eval`) driving #5/#6's generation pipeline into the held-out
eval set: 50 transcripts/category (residents + prospects = 100 total), split ~17/17/16 across
variability tiers per category (then evenly across call reasons within each tier), run through
the quality gates, and written as one flat JSONL file. Generation runs independently of
`build_training_corpus.py` — its own seed, its own persona pool via a fresh `generate_fn` call
per item — and never touches the training corpus's output.

Must never import from `build_training_corpus.py` — the two generate and split independently,
sharing only `corpus_common`, `quality_gates`, and `transcript_gen` (see
`tests/unit/test_import_isolation.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from trellis.generation.corpus_common import (
    GenerateFn,
    allocate_cells_by_tier,
    generate_records,
    record_to_quality_gate_item,
    write_jsonl,
)
from trellis.generation.quality_gates import run_quality_gates
from trellis.generation.transcript_gen import generate_scenario_item
from trellis.schema.loader import load_all_categories
from trellis.settings import settings

TRANSCRIPTS_PER_CATEGORY = 50
# Fixed so which cells get an allocation remainder is reproducible.
SAMPLE_SEED = 800080

DEFAULT_OUT_DIR = Path("data") / "eval_set"


async def build_eval_set(
    *,
    total: int = TRANSCRIPTS_PER_CATEGORY * 2,
    out_dir: Path = DEFAULT_OUT_DIR,
    generate_fn: GenerateFn = generate_scenario_item,
    sample_seed: int = SAMPLE_SEED,
    concurrency: int = settings.generation_concurrency,
) -> int:
    categories = load_all_categories()
    checkpoint_path = out_dir / "checkpoint.jsonl"
    records = await generate_records(
        categories,
        total,
        generate_fn=generate_fn,
        rng=random.Random(sample_seed),
        id_prefix="eval",
        allocate_cells=allocate_cells_by_tier,
        checkpoint_path=checkpoint_path,
        concurrency=concurrency,
    )

    run_quality_gates(
        [record_to_quality_gate_item(r) for r in records],
        diversity_review_path=out_dir / "diversity_review.json",
        injection_review_path=out_dir / "injection_review.json",
        distractor_review_path=out_dir / "distractor_review.json",
    )

    n = write_jsonl(out_dir / "eval.jsonl", records)
    checkpoint_path.unlink(missing_ok=True)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the held-out eval set, independently of the training corpus, "
        "split ~evenly across variability tiers per category."
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=settings.generation_concurrency,
        help="Max concurrent generate_fn calls (default: settings.generation_concurrency).",
    )
    args = parser.parse_args()

    total = args.pilot if args.pilot is not None else TRANSCRIPTS_PER_CATEGORY * 2
    n = asyncio.run(build_eval_set(total=total, out_dir=args.out_dir, concurrency=args.concurrency))
    print(f"eval set: {n} records written to {args.out_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
