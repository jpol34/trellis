"""Remote GPU inference backend: calls a deployed RunPod Serverless endpoint hosting
`TrellisModel` on GPU, instead of running the model in-process (see `TrellisModel`, the local
CPU/GPU backend both implement the same `TrellisBackend` protocol in `trellis_model.py`).

Uses RunPod's submit-then-poll flow (`POST /run` + `GET /status/<job_id>`), not `/runsync` —
`/runsync`'s ~90s server-side wait window doesn't reliably cover a cold start (a fresh worker
constructing `TrellisModel` from scratch), which real-world reports put at 7s-2min+ for a
lazily-loaded model; `/runsync` would silently return `IN_PROGRESS` with no result in that case
rather than failing cleanly.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from trellis.model.trellis_model import DiscriminationResult
from trellis.reference.http import post_with_retry
from trellis.schema.types import FieldSpec
from trellis.settings import settings

_RUNPOD_BASE_URL = "https://api.runpod.ai/v2"


class RunPodJobFailedError(RuntimeError):
    """The RunPod job reported `status == "FAILED"` — RunPod returns this as an HTTP 200 with
    an `error` field (e.g. the handler itself raised, such as a cold-started worker OOMing
    during model construction), not as a 4xx/5xx, so it must be checked via the response body's
    `status` field rather than the HTTP status code."""


class RunPodPollTimeoutError(RuntimeError):
    """The RunPod job didn't reach a terminal status within `trellis_http_poll_timeout_seconds`."""


class TrellisHttpModel:
    """Calls a RunPod Serverless endpoint hosting `TrellisModel` on GPU, implementing the same
    `discriminate_batch` shape the local backend does."""

    def __init__(self, endpoint_id: str) -> None:
        self._endpoint_id = endpoint_id
        # Set after each successful call — the served side's device fallback (GPU OOM -> CPU)
        # is otherwise invisible across the network boundary, unlike the local backend where
        # `TrellisModel.device` is directly inspectable.
        self.last_device_used: str | None = None

    def discriminate_batch(
        self, transcript: str, items: list[tuple[FieldSpec, list[str]]]
    ) -> list[DiscriminationResult | Exception]:
        # `TrellisArm._flush` always calls `discriminate_batch` via `asyncio.to_thread` (see
        # `trellis_arm.py`), so this runs on a worker thread, not the event loop — `asyncio.run`
        # here is safe (there's no already-running loop on this thread to conflict with) and
        # lets the async implementation below reuse `httpx.AsyncClient` +
        # `reference/http.py::post_with_retry`, the same pattern `gpt51_arm.py`/`jev_arm.py` use.
        return asyncio.run(self._discriminate_batch_async(transcript, items))

    async def _discriminate_batch_async(
        self, transcript: str, items: list[tuple[FieldSpec, list[str]]]
    ) -> list[DiscriminationResult | Exception]:
        headers = {
            "Authorization": f"Bearer {settings.runpod_api_key}",
            "content-type": "application/json",
        }
        payload = {
            "input": {
                "transcript": transcript,
                "items": [
                    {"field_name": field.name, "candidates": candidates}
                    for field, candidates in items
                ],
            }
        }

        async with httpx.AsyncClient() as client:
            submit_resp = await post_with_retry(
                client,
                f"{_RUNPOD_BASE_URL}/{self._endpoint_id}/run",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            job_id = submit_resp.json()["id"]
            output = await self._poll_until_done(client, headers, job_id)

        self.last_device_used = output.get("device_used")

        raw_results = output["results"]
        results: list[DiscriminationResult | Exception] = []
        for raw, (_field, candidates) in zip(raw_results, items, strict=True):
            if "error" in raw:
                results.append(ValueError(raw["error"]))
                continue
            chosen_index = raw["chosen_index"]
            if not 0 <= chosen_index < len(candidates):
                results.append(
                    ValueError(
                        f"trellis http backend returned candidate index {chosen_index} out of "
                        f"range for {len(candidates)} candidates"
                    )
                )
                continue
            results.append(
                DiscriminationResult(
                    chosen_index=chosen_index,
                    chosen_value=raw["chosen_value"],
                    confidence=raw["confidence"],
                )
            )
        return results

    async def _poll_until_done(
        self, client: httpx.AsyncClient, headers: dict, job_id: str
    ) -> dict:
        deadline = time.monotonic() + settings.trellis_http_poll_timeout_seconds
        url = f"{_RUNPOD_BASE_URL}/{self._endpoint_id}/status/{job_id}"

        while True:
            resp = await client.get(url, headers=headers, timeout=30.0)
            resp.raise_for_status()
            body = resp.json()
            status = body["status"]

            if status == "COMPLETED":
                return body["output"]
            if status == "FAILED":
                raise RunPodJobFailedError(body.get("error", "unknown error"))
            if time.monotonic() >= deadline:
                raise RunPodPollTimeoutError(
                    f"RunPod job {job_id} did not complete within "
                    f"{settings.trellis_http_poll_timeout_seconds}s (last status: {status!r})"
                )
            await asyncio.sleep(settings.trellis_http_poll_interval_seconds)
