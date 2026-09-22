"""Real token-usage-based $ cost for the arms that are actually billed per token (gpt-5.1, jev).
Adapted from laya-bench's `eval/metrics.py::BackendCost`/`dataset_usage_cost` (same "compute
real cost from a run's own observed usage, not a rate-card estimate" shape), generalized for a
backend whose input and output tokens are priced independently — laya-bench's arms all have free
output, but gpt-5.1's output is priced 8x its input.

trellis is intentionally absent from `BACKEND_COSTS`: its cost is compute (RunPod GPU $/hr or
local CPU), not token-based, and is out of scope for this module. `report.py` renders that as
"n/a" — deliberately not "$0", so it isn't misread as free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCost:
    input_rate_per_mtok: float
    output_rate_per_mtok: float | None  # None means output is free (e.g. jev)
    note: str


def usage_cost(cost: BackendCost, input_tokens: int | None, output_tokens: int | None) -> float:
    """Real $ cost for one call's actual observed token usage. Missing (`None`) token counts
    contribute 0 rather than raising — a call that somehow returned no usage data still has a
    latency/error to report elsewhere, just no cost contribution here."""
    total = (input_tokens or 0) / 1_000_000 * cost.input_rate_per_mtok
    if cost.output_rate_per_mtok is not None:
        total += (output_tokens or 0) / 1_000_000 * cost.output_rate_per_mtok
    return total


BACKEND_COSTS: dict[str, BackendCost] = {
    "gpt-5.1": BackendCost(
        input_rate_per_mtok=1.25,
        output_rate_per_mtok=10.00,
        note="OpenAI gpt-5.1, standard (non-batch) pricing per "
        "developers.openai.com/api/docs/pricing, checked 2026-09-22.",
    ),
    "jev": BackendCost(
        input_rate_per_mtok=0.042,
        output_rate_per_mtok=None,
        note="typesafe.ai Jev, $/MTok input; output tokens are free per typesafe.ai's pricing "
        "(same rate as laya-bench's eval/metrics.py::JEV_COST, dated 2026-09-20).",
    ),
}
