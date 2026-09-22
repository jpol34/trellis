"""RunPod Serverless handler hosting `TrellisModel` on GPU — the served side of the wire
contract `TrellisHttpModel` (`trellis/model/trellis_http_model.py`) speaks as a client.

Follows the `runpod` PyPI package's own worker convention: a `handler(event)` function passed to
`runpod.serverless.start`, which owns the job-polling loop, concurrency, and turning an unhandled
exception raised out of `handler` into a `status: "FAILED"` response with an `error` field — none
of that is reimplemented here.
"""

from __future__ import annotations

import threading

import runpod

from trellis.model.trellis_model import TrellisModel
from trellis.schema.types import FieldSpec
from trellis.settings import settings

_model: TrellisModel | None = None
# Defense-in-depth, not currently load-bearing: RunPod's worker runs one asyncio event loop on
# a single OS thread (`rp_scale.py`'s `run_jobs()` dispatches jobs as `asyncio.create_task`s,
# not real threads), and `handler()` below is a plain synchronous `def`, so it fully blocks that
# loop for its entire duration — no other job's coroutine can interleave with it, meaning two
# `handler()` calls can never actually race on `_model` being `None` as things stand today. Kept
# anyway as double-checked locking in case `handler` is ever converted to `async def` (letting
# jobs genuinely interleave) — if that happens, reconsider this primitive too, since an
# uncontended `threading.Lock.acquire()` inside an `async def` is fine, but a *contended* one
# would itself block the event loop the same way a synchronous `handler()` does now.
_construct_lock = threading.Lock()


def _get_model() -> TrellisModel:
    global _model
    if _model is None:
        with _construct_lock:
            if _model is None:  # re-check: another job may have won the race
                _model = TrellisModel(settings.trellis_checkpoint_path, device="cuda")
    return _model


def _placeholder_field(field_name: str) -> FieldSpec:
    # `TrellisModel.discriminate_batch` only ever reads `field.name` from its `FieldSpec`
    # arguments (it builds each question's instructions string from `.name` alone — see
    # `trellis_model.py`), but the wire payload only carries `field_name`, not a full
    # `FieldSpec`. The other fields are real schema concerns (used elsewhere for scoring/
    # distractor generation) that never reach the served side, so this deliberately fills them
    # with neutral placeholders rather than threading the whole `FieldSpec` over the wire.
    return FieldSpec(
        name=field_name,
        match_type="exact",
        required=False,
        distractor_strategy="none",
    )


def handler(event: dict) -> dict:
    model = _get_model()

    payload = event["input"]
    transcript = payload["transcript"]
    items = [
        (_placeholder_field(item["field_name"]), item["candidates"])
        for item in payload["items"]
    ]

    raw_results = model.discriminate_batch(transcript, items)

    results = []
    for result in raw_results:
        if isinstance(result, Exception):
            results.append({"error": str(result)})
        else:
            results.append(
                {
                    "chosen_index": result.chosen_index,
                    "chosen_value": result.chosen_value,
                    "confidence": result.confidence,
                }
            )

    return {"device_used": model.device, "results": results}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
