"""Reference arm wrapping the CPU-hosted, fine-tuned `TrellisModel` (see #10). Unlike the
gpt-5.1 and jev arms this is an in-process call with no network hop, but `discriminate()` still
runs a blocking forward pass, so it's offloaded to a thread to keep `answer()`'s async contract
honest for callers running arms concurrently.
"""

from __future__ import annotations

import asyncio

from trellis.model.trellis_model import TrellisModel
from trellis.reference.base import ArmAnswer, register_arm
from trellis.schema.types import FieldSpec
from trellis.settings import settings


class TrellisArm:
    name = "trellis"
    mode = "closed_set"

    def __init__(self) -> None:
        # `TrellisModel` construction loads the checkpoint onto `device` — too expensive to pay
        # at import time (when `register_arm` below runs), so it's deferred to the first call.
        self._model: TrellisModel | None = None
        # Guards first-use construction: concurrent `answer()` calls racing on `self._model`
        # being None could otherwise each load a checkpoint and silently discard one.
        self._construct_lock = asyncio.Lock()

    async def _get_model(self) -> TrellisModel:
        if self._model is None:
            async with self._construct_lock:
                if self._model is None:  # re-check: another task may have won the race
                    self._model = TrellisModel(
                        settings.trellis_checkpoint_path, device=settings.trellis_device
                    )
        return self._model

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer:
        if candidates is None:
            raise ValueError("trellis arm is closed_set-only and requires candidates")

        model = await self._get_model()
        result = await asyncio.to_thread(model.discriminate, transcript, field, candidates)

        return ArmAnswer(
            value=result.chosen_value,
            chosen_index=result.chosen_index,
            confidence=result.confidence,
        )


register_arm(TrellisArm())
