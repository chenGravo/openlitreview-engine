from decimal import Decimal

import pytest

from openlitreview.budget import BudgetExceeded, BudgetLedger
from openlitreview.schemas import BudgetSettings


def test_task_reservation_and_actual_reconciliation(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.sqlite", BudgetSettings())
    ledger.reserve_task("task-1", Decimal("15"))
    call_id, estimate = ledger.authorize_call(
        "task-1", "deepseek-v4-pro", input_tokens=1_000_000, max_output_tokens=100_000
    )
    assert estimate == Decimal("3.9600")
    actual = ledger.reconcile_call(call_id, 900_000, 80_000)
    assert actual == Decimal("3.1800")
    summary = ledger.complete_task("task-1")
    assert summary["actual_cny"] == "3.1800"


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

