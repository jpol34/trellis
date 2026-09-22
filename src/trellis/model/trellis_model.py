"""Synchronous CPU inference boundary over a fine-tuned laya checkpoint.

Mirrors laya-bench's `backend.py::LayaBackend` shape (load once per process, then run
blocking `infer`-style calls) so a future served-gateway wrapper can reuse this module without
rework, even though no such gateway exists in this project yet.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

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


class TrellisBackend(Protocol):
    """Anything that can run `discriminate_batch` for `TrellisArm` — locally on CPU/GPU
    (`TrellisModel`) or against a remote served GPU (`TrellisHttpModel`)."""

    def discriminate_batch(
        self, transcript: str, items: list[tuple[FieldSpec, list[str]]]
    ) -> list[DiscriminationResult | Exception]: ...


def _question_key(i: int, chunk_size: int) -> str:
    # A chunk of exactly one item is what `discriminate()` sends, so it keeps the original
    # unsuffixed key — the exact wire shape a single-item call sent before batching existed.
    if chunk_size == 1:
        return QUESTION_NAME
    return f"{QUESTION_NAME}_{i}"


def _parse_choice_answer(
    answers: dict, key: str, candidates: list[str]
) -> DiscriminationResult:
    try:
        answer = answers[key]
    except KeyError as e:
        raise ValueError(
            f"trellis model response is missing the {key!r} answer: {answers!r}"
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


class TrellisModel:
    """Wraps a `laya.Agent` loaded from a fine-tuned checkpoint directory on `device`. Construct
    once per process — loading places the checkpoint on `device`, too expensive to repeat per
    call."""

    def __init__(self, checkpoint_dir: str, device: str = "cpu") -> None:
        self._agent = laya.load(checkpoint_dir, device=device)
        # Guards each `discriminate_batch()` call: `torch.set_num_threads` is a process-global
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

    def discriminate_batch(
        self, transcript: str, items: list[tuple[FieldSpec, list[str]]]
    ) -> list[DiscriminationResult | Exception]:
        """Runs one or more forward passes choosing among each item's candidates for its field,
        against the same transcript. Batches items together into `system_one()` calls of at most
        `settings.trellis_batch_max_size`, one bad item's failure never affecting siblings in the
        same chunk. Results are returned in the same order as `items`."""
        results: list[DiscriminationResult | Exception | None] = [None] * len(items)

        valid_indices: list[int] = []
        for i, (_field, candidates) in enumerate(items):
            if not candidates:
                results[i] = ValueError("discriminate() requires a non-empty candidates list")
            else:
                valid_indices.append(i)

        max_size = settings.trellis_batch_max_size
        for chunk_start in range(0, len(valid_indices), max_size):
            chunk_indices = valid_indices[chunk_start : chunk_start + max_size]
            chunk_size = len(chunk_indices)
            questions = {
                _question_key(i, chunk_size): {
                    "type": "choice",
                    "instructions": f"Which value applies for the {items[i][0].name!r} field?",
                    "criteria": candidates_to_criteria(items[i][1]),
                }
                for i in chunk_indices
            }

            # Serialized: `system_one` on the shared `laya.Agent` plus the device-resolution/
            # thread-pinning that follows it are not safe to run concurrently from multiple
            # threads against one `TrellisModel` instance. A chunk's `system_one` failure is
            # caught here (rather than left to propagate out of the whole method) so it can
            # never discard results already computed by an earlier, successful chunk.
            try:
                with self._lock:
                    response = self._agent.system_one(transcript, questions)
                    # `system_one` can itself fall back to CPU mid-call (e.g. a GPU OOM after
                    # construction), so re-resolve/re-pin after every call rather than trusting
                    # the construction-time value.
                    self.device = self._resolve_device_and_pin_threads()
            except Exception as e:  # noqa: BLE001 - isolated to this chunk's items only
                for i in chunk_indices:
                    results[i] = e
                continue

            answers = response["answers"]
            for i in chunk_indices:
                try:
                    key = _question_key(i, chunk_size)
                    results[i] = _parse_choice_answer(answers, key, items[i][1])
                except Exception as e:  # noqa: BLE001 - one item's malformed answer isolated
                    results[i] = e

        return results  # type: ignore[return-value]

    def discriminate(
        self, transcript: str, field: FieldSpec, candidates: list[str]
    ) -> DiscriminationResult:
        """Runs one synchronous forward pass choosing among `candidates` for `field`. Thin
        wrapper over `discriminate_batch()` for a single item."""
        result = self.discriminate_batch(transcript, [(field, candidates)])[0]
        if isinstance(result, Exception):
            raise result
        return result
