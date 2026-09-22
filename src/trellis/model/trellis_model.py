"""Synchronous CPU inference boundary over a fine-tuned laya checkpoint.

Mirrors laya-bench's `backend.py::LayaBackend` shape (load once per process, then run
blocking `infer`-style calls) so a future served-gateway wrapper can reuse this module without
rework, even though no such gateway exists in this project yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import laya
import torch

from trellis.reference.wire import candidates_to_criteria, criteria_key_to_index
from trellis.schema.types import FieldSpec
from trellis.settings import settings

QUESTION_NAME = "field"


@dataclass(frozen=True)
class DiscriminationResult:
    chosen_index: int
    chosen_value: str
    confidence: float


class TrellisModel:
    """Wraps a `laya.Agent` loaded from a fine-tuned checkpoint directory on `device`. Construct
    once per process — loading places the checkpoint on `device`, too expensive to repeat per
    call."""

    def __init__(self, checkpoint_dir: str, device: str = "cpu") -> None:
        self._agent = laya.load(checkpoint_dir, device=device)
        # `laya.Agent` silently falls back to CPU (only printing a warning) if the requested
        # device is unavailable or OOMs, so the thread-pinning decision below must key off the
        # *resolved* device, not the requested one.
        self.device = self._agent.device.type

        if self.device == "cpu":
            # PyTorch sizes its default thread pool from the host machine's core count, not a
            # per-request budget — on a shared/containerized host that causes contention well
            # past what a single sequential forward pass over this checkpoint needs. See
            # laya-bench's `backend.py::LayaBackend` for the same rationale.
            torch.set_num_threads(settings.trellis_cpu_threads)

    def discriminate(
        self, transcript: str, field: FieldSpec, candidates: list[str]
    ) -> DiscriminationResult:
        """Runs one synchronous forward pass choosing among `candidates` for `field`."""
        criteria = candidates_to_criteria(candidates)
        response = self._agent.system_one(
            transcript,
            {
                QUESTION_NAME: {
                    "type": "choice",
                    "instructions": f"Which value applies for the {field.name!r} field?",
                    "criteria": criteria,
                }
            },
        )

        try:
            answer = response["answers"][QUESTION_NAME]
        except KeyError as e:
            raise ValueError(
                f"trellis model response is missing the {QUESTION_NAME!r} answer: "
                f"{response.get('answers')!r}"
            ) from e

        chosen_index = criteria_key_to_index(answer["choice"])
        if not 0 <= chosen_index < len(candidates):
            raise ValueError(
                f"trellis model returned candidate index {chosen_index} out of range for "
                f"{len(candidates)} candidates: {answer['choice']!r}"
            )

        return DiscriminationResult(
            chosen_index=chosen_index,
            chosen_value=candidates[chosen_index],
            confidence=answer["confidence"],
        )
