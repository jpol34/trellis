from __future__ import annotations

from trellis.validate.costs import BACKEND_COSTS, BackendCost, usage_cost

_BOTH_PRICED = BackendCost(input_rate_per_mtok=1.25, output_rate_per_mtok=10.00, note="test")
_OUTPUT_FREE = BackendCost(input_rate_per_mtok=0.042, output_rate_per_mtok=None, note="test")


def test_usage_cost_prices_both_input_and_output_when_both_rates_set() -> None:
    cost = usage_cost(_BOTH_PRICED, input_tokens=1_000_000, output_tokens=100_000)
    assert cost == 1.25 + 1.00


def test_usage_cost_ignores_output_tokens_when_output_rate_is_none() -> None:
    cost = usage_cost(_OUTPUT_FREE, input_tokens=1_000_000, output_tokens=999_999_999)
    assert cost == 0.042


def test_usage_cost_treats_none_tokens_as_zero() -> None:
    assert usage_cost(_BOTH_PRICED, input_tokens=None, output_tokens=None) == 0.0
    assert usage_cost(_OUTPUT_FREE, input_tokens=None, output_tokens=None) == 0.0


def test_backend_costs_registry_has_gpt51_and_jev_not_trellis() -> None:
    assert "gpt-5.1" in BACKEND_COSTS
    assert "jev" in BACKEND_COSTS
    assert "trellis" not in BACKEND_COSTS
