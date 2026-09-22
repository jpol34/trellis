"""Partial live validation of #15's `trellis-eval` pipeline: the `trellis` arm needs only the
local fine-tuned checkpoint (no external API credentials), so it can run end-to-end against a
slice of the real held-out eval set. Skipped when either the checkpoint or the real eval set
isn't present on disk, mirroring `test_trellis_arm.py`'s skip-if-absent pattern — this is not a
substitute for a full three-arm live run (that also needs OPENAI_API_KEY/TYPESAFE_API_KEY, see
`scripts/`), only evidence the pipeline itself (loading, dispatch, scoring, report rendering)
works end-to-end against real data for the one arm that doesn't need external credentials."""

from __future__ import annotations

from pathlib import Path

import pytest

import trellis.reference.trellis_arm  # noqa: F401 - import registers "trellis" into ARMS
from trellis.reference.base import ARMS
from trellis.settings import settings
from trellis.validate.report import render_markdown
from trellis.validate.run_eval import run_eval

CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "run-20260922T062830Z"
)
EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_set"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR.exists() or not any(EVAL_DIR.glob("*.jsonl")),
    reason=f"checkpoint ({CHECKPOINT_DIR}) or eval set ({EVAL_DIR}) not present",
)


@pytest.fixture(autouse=True)
def _checkpoint_settings(monkeypatch):
    monkeypatch.setattr(settings, "trellis_checkpoint_path", str(CHECKPOINT_DIR))
    monkeypatch.setattr(settings, "trellis_device", "cpu")


async def test_trellis_arm_runs_clean_against_real_eval_set_slice() -> None:
    trellis_arm = ARMS["trellis"]

    results, categories = await run_eval(
        eval_dir=EVAL_DIR, arms={"trellis": trellis_arm}, concurrency=1, limit=6
    )

    assert categories
    dataset_results = results["trellis"].datasets
    assert dataset_results  # at least one category scored

    all_items = [item for ds in dataset_results.values() for item in ds.items]
    assert all_items
    errors = [item for item in all_items if item.error is not None]
    assert not errors, f"trellis arm calls failed unexpectedly: {[e.error for e in errors]}"

    for item in all_items:
        assert item.state in (
            "present_correct",
            "present_incorrect",
            "hallucinated",
            "silently_dropped",
            "correctly_absent",
        )

    report = render_markdown(results, categories)
    assert "trellis" in report
    assert "Different tasks, not a ranked leaderboard" in report
