"""Loads the real fine-tuned checkpoint and runs it on CPU — slow (full model load) and
requires the checkpoint to be present on disk, so this lives in integration rather than unit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trellis.model.trellis_model import TrellisModel
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


@pytest.fixture(scope="module")
def model() -> TrellisModel:
    return TrellisModel(str(CHECKPOINT_DIR), device="cpu")


def test_loads_on_cpu(model: TrellisModel):
    assert model.device == "cpu"


def test_cpu_threads_pinned_to_settings(model: TrellisModel):
    # Asserts against the *actual* torch thread count post-construction rather than trusting
    # that `torch.set_num_threads` was merely called.
    assert torch.get_num_threads() == settings.trellis_cpu_threads


def test_discriminate_returns_correct_answer_for_a_clear_transcript(model: TrellisModel):
    result = model.discriminate(TRANSCRIPT, FIELD, CANDIDATES)

    assert result.chosen_index == 0
    assert result.chosen_value == "leak"
    assert result.confidence > 0.5
