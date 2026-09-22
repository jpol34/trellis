"""Statistical methodology shared by both scoring modes (closed_set.py and matchers.py-based
open extraction), operating on the common `FiveState` produced by `states.resolve_state`.
`bootstrap_accuracy_ci` and `latency_percentiles` are adapted from laya-bench's
`eval/metrics.py` (same resampling/percentile logic); `expected_calibration_error` is
deliberately not carried over — it doesn't map cleanly onto either of trellis's two scoring
modes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from trellis.validate.states import FiveState


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float


def bootstrap_accuracy_ci(
    correct: list[bool],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap CI on accuracy: resample `correct` with replacement `n_resamples` times, take
    the percentile interval of the resample means. A bare point estimate on small eval sets can
    look decisive when it's actually noise."""
    if not correct:
        return ConfidenceInterval(point=float("nan"), low=float("nan"), high=float("nan"))
    n = len(correct)
    point = sum(correct) / n
    rng = random.Random(seed)
    resample_means = sorted(
        sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    alpha = 1 - confidence
    low_idx = max(0, int((alpha / 2) * n_resamples))
    high_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)
    return ConfidenceInterval(
        point=point, low=resample_means[low_idx], high=resample_means[high_idx]
    )


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    ordered = sorted(latencies_ms)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
        return ordered[idx]

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


def precision_recall_f1(states: list[FiveState]) -> PRF1:
    """Precision/recall over the 5-state resolution, framed around the "does the model claim a
    value that isn't there" vs. "does the model get the value right when there is one"
    distinction:

    - TP = `present_correct` (a real value, extracted correctly).
    - FP = `hallucinated` (claimed a value on a field that's genuinely absent) — precision
      penalizes false claims of presence, not wrong-value substitutions.
    - FN = `silently_dropped` + `present_incorrect` (a real value existed but the model either
      said nothing or got it wrong) — recall penalizes any failure to correctly surface a value
      that was there, by omission or by mistake.
    - `correctly_absent` is a true negative and doesn't enter precision/recall.

    Precision/recall/F1 are 0.0 when their denominator is 0 (no positive predictions / no
    actual positives), rather than raising or returning NaN.
    """
    tp = sum(1 for s in states if s == "present_correct")
    fp = sum(1 for s in states if s == "hallucinated")
    fn = sum(1 for s in states if s in ("silently_dropped", "present_incorrect"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return PRF1(precision=precision, recall=recall, f1=f1)
