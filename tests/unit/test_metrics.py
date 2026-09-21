from __future__ import annotations

import pytest

from trellis.validate.metrics import (
    ConfidenceInterval,
    bootstrap_accuracy_ci,
    latency_percentiles,
    precision_recall_f1,
)


def test_bootstrap_accuracy_ci_shape() -> None:
    result = bootstrap_accuracy_ci([True, True, False, True, False], seed=0)
    assert isinstance(result, ConfidenceInterval)
    assert result.point == pytest.approx(0.6)
    assert result.low <= result.point <= result.high


def test_bootstrap_accuracy_ci_deterministic_given_seed() -> None:
    correct = [True, True, False, True, False, True, False, True, True, False]
    first = bootstrap_accuracy_ci(correct, n_resamples=500, confidence=0.95, seed=42)
    second = bootstrap_accuracy_ci(correct, n_resamples=500, confidence=0.95, seed=42)
    assert first == second


def test_bootstrap_accuracy_ci_different_seed_can_differ() -> None:
    correct = [True, False, True, False, True, False, True, False]
    a = bootstrap_accuracy_ci(correct, n_resamples=500, seed=1)
    b = bootstrap_accuracy_ci(correct, n_resamples=500, seed=2)
    assert a.point == b.point  # point estimate doesn't depend on resampling
    # bounds are resampling-dependent and not required to match across seeds


def test_bootstrap_accuracy_ci_empty() -> None:
    result = bootstrap_accuracy_ci([])
    assert result.point != result.point  # nan


def test_latency_percentiles_shape() -> None:
    result = latency_percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert set(result.keys()) == {"p50", "p95", "p99"}


def test_latency_percentiles_deterministic() -> None:
    latencies = [120.0, 80.0, 300.0, 150.0, 90.0, 400.0, 110.0, 130.0, 95.0, 500.0]
    first = latency_percentiles(latencies)
    second = latency_percentiles(latencies)
    assert first == second


def test_latency_percentiles_known_values() -> None:
    # ordered: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    latencies = [100.0, 90.0, 10.0, 20.0, 80.0, 30.0, 70.0, 40.0, 60.0, 50.0]
    result = latency_percentiles(latencies)
    assert result["p50"] == 50.0
    assert result["p95"] == 100.0
    assert result["p99"] == 100.0


def test_latency_percentiles_empty() -> None:
    result = latency_percentiles([])
    assert result["p50"] != result["p50"]  # nan


# precision_recall_f1: hand-built fixture.
# states = 3 present_correct, 2 present_incorrect, 1 silently_dropped, 2 hallucinated,
#          4 correctly_absent (ignored by precision/recall)
# tp = 3 (present_correct)
# fp = 2 (hallucinated)
# fn = 2 (present_incorrect) + 1 (silently_dropped) = 3
# precision = tp / (tp + fp) = 3 / 5 = 0.6
# recall    = tp / (tp + fn) = 3 / 6 = 0.5
# f1        = 2 * 0.6 * 0.5 / (0.6 + 0.5) = 0.6 / 1.1 = 0.5454545454545454
FIXTURE_STATES = (
    ["present_correct"] * 3
    + ["present_incorrect"] * 2
    + ["silently_dropped"] * 1
    + ["hallucinated"] * 2
    + ["correctly_absent"] * 4
)


def test_precision_recall_f1_hand_worked_fixture() -> None:
    result = precision_recall_f1(FIXTURE_STATES)
    assert result.precision == pytest.approx(0.6)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5454545454545454)


def test_precision_recall_f1_all_correct() -> None:
    result = precision_recall_f1(["present_correct", "present_correct", "correctly_absent"])
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_precision_recall_f1_no_positives() -> None:
    # no present_correct, no hallucinated, no present_incorrect/silently_dropped:
    # tp=fp=fn=0, so precision/recall/f1 all default to 0.0 rather than dividing by zero.
    result = precision_recall_f1(["correctly_absent", "correctly_absent"])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_precision_recall_f1_empty() -> None:
    result = precision_recall_f1([])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
