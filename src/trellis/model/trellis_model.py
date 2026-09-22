"""Synchronous CPU inference boundary over a fine-tuned laya checkpoint.

Mirrors laya-bench's `backend.py::LayaBackend` shape (load once per process, then run
blocking `infer`-style calls) so a future served-gateway wrapper can reuse this module without
rework, even though no such gateway exists in this project yet.
"""

from __future__ import annotations

import threading
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
        # Guards each `discriminate()` call: `torch.set_num_threads` is a process-global
        # setting and `self.device` is a plain shared attribute, so concurrent calls on this
        # instance (e.g. via `asyncio.to_thread` from a shared arm) must be serialized rather
        # than racing each other's device-resolution/thread-pinning.
        self._lock = threading.Lock()
        self.device = self._resolve_device_and_pin_threads()

    def _resolve_device_and_pin_threads(self) -> str:
        # `laya.Agent` can silently fall back to CPU (only printing a warning) both at load time
        # and mid-call, on OOM, so the thread-pinning decision must key off the *resolved* device
        # at the time of the call, not a value cached from construction.
        device = self._agent.device.type
        if device == "cpu":
            # PyTorch sizes its default thread pool from the host machine's core count, not a
            # per-request budget — on a shared/containerized host that causes contention well
            # past what a single sequential forward pass over this checkpoint needs. See
            # laya-bench's `backend.py::LayaBackend` for the same rationale.
            torch.set_num_threads(settings.trellis_cpu_threads)
        return device

    def discriminate(
        self, transcript: str, field: FieldSpec, candidates: list[str]
    ) -> DiscriminationResult:
        """Runs one synchronous forward pass choosing among `candidates` for `field`."""
        if not candidates:
            raise ValueError("discriminate() requires a non-empty candidates list")

        criteria = candidates_to_criteria(candidates)

        # Serialized: `system_one` on the shared `laya.Agent` plus the device-resolution/
        # thread-pinning that follows it are not safe to run concurrently from multiple
        # threads against one `TrellisModel` instance.
        with self._lock:
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
            # `system_one` can itself fall back to CPU mid-call (e.g. a GPU OOM after
            # construction), so re-resolve/re-pin after every call rather than trusting the
            # construction-time value.
            self.device = self._resolve_device_and_pin_threads()

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
