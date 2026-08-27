from decimal import Decimal

import pytest

from openlitreview.budget import BudgetExceeded, BudgetLedger
from openlitreview.pricing import get_price
from openlitreview.schemas import BudgetSettings


def test_task_reservation_and_actual_reconciliation(tmp_path) -> None:
    ledger = BudgetLedger(
        tmp_path / "budget.sqlite", BudgetSettings(single_request_cap_cny=10)
    )
    ledger.reserve_task("task-1", Decimal("15"))
    call_id, estimate = ledger.authorize_call(
        "task-1", "deepseek-v4-pro", input_tokens=1_000_000, max_output_tokens=100_000
    )
    assert estimate == Decimal("5.1700")
    actual = ledger.reconcile_call(call_id, 900_000, 80_000)
    assert actual == Decimal("4.1600")
    summary = ledger.complete_task("task-1")
    assert summary["actual_cny"] == "4.1600"


def test_monthly_hard_stop_blocks_new_task(tmp_path) -> None:
    settings = BudgetSettings(task_reservation_cny=30)
    ledger = BudgetLedger(tmp_path / "budget.sqlite", settings)
    ledger.reserve_task("task-1", Decimal("30"))
    ledger.reserve_task("task-2", Decimal("30"))
    ledger.reserve_task("task-3", Decimal("30"))
    with pytest.raises(BudgetExceeded, match="hard stop"):
        ledger.reserve_task("task-4", Decimal("1"))


def test_unknown_model_is_blocked(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.sqlite", BudgetSettings())
    ledger.reserve_task("task-1")
    with pytest.raises(ValueError, match="Unknown or unpriced"):
        ledger.authorize_call("task-1", "unreviewed-model", 100, 100)


def test_kimi_prices_use_conservative_cny_planning_values() -> None:
    price = get_price("kimi-k2.6")
    assert price.input_cny_per_million == Decimal("8")
    assert price.output_cny_per_million == Decimal("32")


def test_each_model_has_an_independent_ten_cny_task_cap(tmp_path) -> None:
    settings = BudgetSettings(
        task_reservation_cny=30,
        single_request_cap_cny=5,
        per_model_task_cap_cny=10,
        monthly_per_model_cap_cny=10,
    )
    ledger = BudgetLedger(tmp_path / "budget.sqlite", settings)
    ledger.reserve_task("benchmark", Decimal("30"))

    for _ in range(2):
        ledger.authorize_call(
            "benchmark",
            "deepseek-v4-pro",
            input_tokens=1_000_000,
            max_output_tokens=0,
        )
    with pytest.raises(BudgetExceeded, match="Per-model/provider task cap"):
        ledger.authorize_call(
            "benchmark",
            "deepseek-v4-pro",
            input_tokens=300_000,
            max_output_tokens=0,
        )

    ledger.reserve_task("benchmark-retry", Decimal("30"))
    with pytest.raises(BudgetExceeded, match="Monthly per-model/provider cap"):
        ledger.authorize_call(
            "benchmark-retry",
            "deepseek-v4-pro",
            input_tokens=300_000,
            max_output_tokens=0,
        )

    _, estimate = ledger.authorize_call(
        "benchmark",
        "kimi-k2.6",
        input_tokens=500_000,
        max_output_tokens=0,
    )
    assert estimate == Decimal("4.4000")


def test_switching_alias_cannot_bypass_provider_cap(tmp_path) -> None:
    settings = BudgetSettings(
        task_reservation_cny=30,
        single_request_cap_cny=1,
        per_model_task_cap_cny=1,
        monthly_per_model_cap_cny=1,
    )
    ledger = BudgetLedger(tmp_path / "budget.sqlite", settings)
    ledger.reserve_task("benchmark", Decimal("30"))
    ledger.authorize_call(
        "benchmark",
        "deepseek-v4-pro",
        input_tokens=225_000,
        max_output_tokens=0,
    )
    with pytest.raises(BudgetExceeded, match="Per-model/provider task cap"):
        ledger.authorize_call(
            "benchmark",
            "deepseek-v4-flash",
            input_tokens=100_000,
            max_output_tokens=0,
        )
