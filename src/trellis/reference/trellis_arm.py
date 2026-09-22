"""Reference arm wrapping the CPU-hosted, fine-tuned `TrellisModel` (see #10). Unlike the
gpt-5.1 and jev arms this is an in-process call with no network hop, but `discriminate_batch()`
still runs a blocking forward pass, so it's offloaded to a thread to keep `answer()`'s async
contract honest for callers running arms concurrently.

Sibling `answer()` calls for the same transcript (different fields of the same record) are
coalesced into one `discriminate_batch()` call rather than each paying for its own forward pass:
the eval runner (`run_eval.py`) creates every field-task for a record before starting the next
one, so by the time the first task actually runs, its siblings are reliably already scheduled.
The first `answer()` call for a transcript becomes the "leader" — it briefly debounces to let
those siblings register themselves, then flushes the whole batch in one model call and resolves
each caller's own `Future`.
"""

from __future__ import annotations

import asyncio

from trellis.model.trellis_model import TrellisModel
from trellis.reference.base import ArmAnswer, register_arm
from trellis.schema.types import FieldSpec
from trellis.settings import settings


class _PendingItem:
    __slots__ = ("field", "candidates", "future")

    def __init__(
        self, field: FieldSpec, candidates: list[str], future: asyncio.Future
    ) -> None:
        self.field = field
        self.candidates = candidates
        self.future = future


class TrellisArm:
    name = "trellis"
    mode = "closed_set"
    # `TrellisModel.discriminate_batch` serializes internally on a lock (see #11's review), and
    # sibling field-calls for one record are coalesced into a single batched call (see module
    # docstring), so a higher cap here lets a full record's worth of fields (checked: both
    # category configs define 9 fields) register as siblings before the leader flushes, instead
    # of queueing them behind each other one at a time.
    max_concurrency = 16

    def __init__(self) -> None:
        # `TrellisModel` construction loads the checkpoint onto `device` — too expensive to pay
        # at import time (when `register_arm` below runs), so it's deferred to the first call.
        self._model: TrellisModel | None = None
        # Guards first-use construction: concurrent `answer()` calls racing on `self._model`
        # being None could otherwise each load a checkpoint and silently discard one.
        self._construct_lock = asyncio.Lock()
        # Per-transcript buffer of not-yet-flushed requests. The check-then-append in `answer()`
        # below has no `await` between reading and mutating this dict, so it's race-free without
        # a lock: nothing else can interleave there.
        self._pending: dict[str, list[_PendingItem]] = {}

    async def _get_model(self) -> TrellisModel:
        if self._model is None:
            async with self._construct_lock:
                if self._model is None:  # re-check: another task may have won the race
                    self._model = TrellisModel(
                        settings.trellis_checkpoint_path, device=settings.trellis_device
                    )
        return self._model

    async def _flush(self, transcript: str, items: list[_PendingItem]) -> None:
        # `_get_model()` is inside this try, not before it: a checkpoint-load failure there is
        # just as fatal to every waiting follower as a `discriminate_batch` failure is, and must
        # resolve their Futures the same way rather than propagating out and leaving them
        # hanging on `await future` forever (`run_eval.py` has no timeout around `answer()`).
        try:
            model = await self._get_model()
            results = await asyncio.to_thread(
                model.discriminate_batch,
                transcript,
                [(item.field, item.candidates) for item in items],
            )
        except Exception as e:  # noqa: BLE001 - every waiting Future must be resolved
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)
            return

        for item, result in zip(items, results, strict=True):
            if item.future.done():
                continue
            if isinstance(result, Exception):
                item.future.set_exception(result)
            else:
                item.future.set_result(result)

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer:
        if candidates is None:
            raise ValueError("trellis arm is closed_set-only and requires candidates")

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        item = _PendingItem(field, candidates, future)

        bucket = self._pending.get(transcript)
        is_leader = bucket is None
        if bucket is None:
            bucket = []
            self._pending[transcript] = bucket
        bucket.append(item)

        if is_leader:
            await asyncio.sleep(settings.trellis_batch_debounce_seconds)
            # Pop the whole buffer before doing anything else, so a failure in `_flush` can
            # never leave stale entries for a future call on this transcript to append onto.
            flushed = self._pending.pop(transcript, [])
            await self._flush(transcript, flushed)

        result = await future

        return ArmAnswer(
            value=result.chosen_value,
            chosen_index=result.chosen_index,
            confidence=result.confidence,
        )


register_arm(TrellisArm())
