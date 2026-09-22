"""Loads the real fine-tuned checkpoint through `TrellisArm` (rather than `TrellisModel`
directly, which `test_trellis_model.py` already covers) — slow and requires the checkpoint to
be present on disk, so this lives in integration rather than unit."""

from __future__ import annotations

from pathlib import Path

import pytest

from trellis.reference.base import ARMS
from trellis.schema.types import FieldSpec
from trellis.settings import settings

CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "run-20260922T062830Z"
)

FIELD = FieldSpec(
    name="issue_type",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="issue_types",
)

TRANSCRIPT = (
    "Resident: Hi, I'm calling about unit 4B. There's water leaking from the ceiling in "
    "my kitchen and it's getting worse. Agent: I'm sorry to hear that, let's get someone "
    "out to look at the leak today."
)
CANDIDATES = ["leak", "noise complaint", "pest control", "no heat", "not mentioned"]

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_DIR.exists(), reason=f"checkpoint not present at {CHECKPOINT_DIR}"
)


@pytest.fixture(autouse=True)
def _checkpoint_settings(monkeypatch):
    monkeypatch.setattr(settings, "trellis_checkpoint_path", str(CHECKPOINT_DIR))
    monkeypatch.setattr(settings, "trellis_device", "cpu")


async def test_trellis_arm_answers_correctly_via_trellis_model():
    arm = ARMS["trellis"]
    assert arm.name == "trellis"

    result = await arm.answer(TRANSCRIPT, FIELD, CANDIDATES)

    assert result["value"] == "leak"
    assert result["chosen_index"] == 0
    assert result["confidence"] > 0.5
