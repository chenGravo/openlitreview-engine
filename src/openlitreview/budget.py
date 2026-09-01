from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from decimal import ROUND_UP, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from .pricing import get_price
from .schemas import BudgetSettings

SHANGHAI = ZoneInfo("Asia/Shanghai")


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: str | Path, settings: BudgetSettings) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self._initialize()

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS task_budget (
                    task_id TEXT PRIMARY KEY,
                    budget_group_id TEXT NOT NULL DEFAULT '',
                    month TEXT NOT NULL,
                    reserved_cny TEXT NOT NULL,
                    actual_cny TEXT NOT NULL DEFAULT '0',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_call (
                    call_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_alias TEXT NOT NULL,
                    estimated_cny TEXT NOT NULL,
                    actual_cny TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES task_budget(task_id)
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(task_budget)").fetchall()
            }
            if "budget_group_id" not in columns:
                connection.execute(
                    "ALTER TABLE task_budget ADD COLUMN budget_group_id TEXT NOT NULL DEFAULT ''"
                )
            rows_without_group = connection.execute(
                "SELECT task_id FROM task_budget WHERE budget_group_id = ''"
            ).fetchall()
            for row in rows_without_group:
                connection.execute(
                    "UPDATE task_budget SET budget_group_id = ? WHERE task_id = ?",
                    (_budget_group_id(row["task_id"]), row["task_id"]),
                )
            legacy_failures = connection.execute(
                "SELECT task_id FROM task_budget WHERE status = 'failed_reserved'"
            ).fetchall()
            for row in legacy_failures:
                actual = self._task_actual(connection, row["task_id"])
                connection.execute(
                    """
                    UPDATE task_budget
                    SET actual_cny = ?, status = 'failed', updated_at = ?
                    WHERE task_id = ?
                    """,
                    (str(actual), _now(), row["task_id"]),
                )

    def reserve_task(self, task_id: str, amount_cny: Decimal | None = None) -> dict[str, str]:
        amount = amount_cny or Decimal(str(self.settings.task_reservation_cny))
        budget_group_id = _budget_group_id(task_id)
        month = _month()
        now = _now()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM task_budget WHERE task_id = ?", (task_id,)
            ).fetchone()
            if existing:
                return dict(existing)
            used = self._month_reserved_or_spent(connection, month)
            hard_stop = Decimal(str(self.settings.monthly_hard_stop_cny))
            if not self.settings.quality_trial_unlimited and used + amount > hard_stop:
                raise BudgetExceeded(
                    f"Monthly internal hard stop would be exceeded: {used} + {amount} > {hard_stop}"
                )
            connection.execute(
                """
                INSERT INTO task_budget
                    (task_id, budget_group_id, month, reserved_cny, actual_cny, status,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, '0', 'reserved', ?, ?)
                """,
                (task_id, budget_group_id, month, str(amount), now, now),
            )
        return self.task_summary(task_id)

    def authorize_call(
        self,
        task_id: str,
        model_alias: str,
        input_tokens: int,
        max_output_tokens: int,
        uncertainty_multiplier: Decimal = Decimal("1.10"),
    ) -> tuple[str, Decimal]:
        price = get_price(model_alias)
        estimated = (
            price.estimate(input_tokens, max_output_tokens) * uncertainty_multiplier
        ).quantize(Decimal("0.0001"), rounding=ROUND_UP)
        single_cap = Decimal(str(self.settings.single_request_cap_cny))
        if not self.settings.quality_trial_unlimited and estimated > single_cap:
            raise BudgetExceeded(f"Single request estimate {estimated} exceeds cap {single_cap}")
        now = _now()
        call_id = uuid.uuid4().hex
        with self._transaction() as connection:
            task = connection.execute(
                "SELECT * FROM task_budget WHERE task_id = ?", (task_id,)
            ).fetchone()
            if not task:
                raise BudgetExceeded("Task budget must be reserved before any model request")
            if task["status"] not in {"reserved", "running"}:
                raise BudgetExceeded(f"Task budget is not active: {task['status']}")
            budget_group_id = task["budget_group_id"] or _budget_group_id(task_id)
            consumed = self._budget_group_calls_reserved_or_spent(
                connection, budget_group_id
            )
            reserved = Decimal(task["reserved_cny"])
            if (
                not self.settings.quality_trial_unlimited
                and consumed + estimated > reserved
            ):
                raise BudgetExceeded(
                    f"Task reservation would be exceeded: {consumed} + {estimated} > {reserved}"
                )
            model_consumed = self._budget_group_provider_calls_reserved_or_spent(
                connection, budget_group_id, price.provider
            )
            model_cap = Decimal(str(self.settings.provider_task_cap_cny(price.provider)))
            if (
                not self.settings.quality_trial_unlimited
                and model_consumed + estimated > model_cap
            ):
                raise BudgetExceeded(
                    "Per-model/provider task cap would be exceeded for "
                    f"{price.provider}: {model_consumed} + {estimated} > {model_cap}"
                )
            monthly_model_consumed = self._month_provider_calls_reserved_or_spent(
                connection, task["month"], price.provider
            )
            monthly_model_cap = Decimal(str(self.settings.monthly_per_model_cap_cny))
            if (
                not self.settings.quality_trial_unlimited
                and monthly_model_consumed + estimated > monthly_model_cap
            ):
                raise BudgetExceeded(
                    "Monthly per-model/provider cap would be exceeded for "
                    f"{price.provider}: {monthly_model_consumed} + {estimated} "
                    f"> {monthly_model_cap}"
                )
            connection.execute(
                """
                INSERT INTO model_call
                    (call_id, task_id, provider, model_alias, estimated_cny, status,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?)
                """,
                (
                    call_id,
                    task_id,
                    price.provider,
                    model_alias,
                    str(estimated),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE task_budget SET status = 'running', updated_at = ? WHERE task_id = ?",
                (now, task_id),
            )
        return call_id, estimated

    def reconcile_call(
        self,
        call_id: str,
        input_tokens: int,
        output_tokens: int,
        *,
        status: str = "completed",
    ) -> Decimal:
        if status not in {"completed", "failed_billed", "failed_unknown"}:
            raise ValueError(f"Unsupported call status: {status}")
        with self._transaction() as connection:
            call = connection.execute(
                "SELECT * FROM model_call WHERE call_id = ?", (call_id,)
            ).fetchone()
            if not call:
                raise KeyError(f"Unknown call id: {call_id}")
            if status == "failed_unknown":
                actual = Decimal(call["estimated_cny"])
            else:
                price = get_price(call["model_alias"])
                actual = price.estimate(input_tokens, output_tokens)
            connection.execute(
                """
                UPDATE model_call
                SET actual_cny = ?, input_tokens = ?, output_tokens = ?, status = ?, updated_at = ?
                WHERE call_id = ?
                """,
                (str(actual), input_tokens, output_tokens, status, _now(), call_id),
            )
        return actual

    def complete_task(self, task_id: str) -> dict[str, str]:
        with self._transaction() as connection:
            actual = self._task_actual(connection, task_id)
            connection.execute(
                """
                UPDATE task_budget
                SET actual_cny = ?, status = 'completed', updated_at = ?
                WHERE task_id = ?
                """,
                (str(actual), _now(), task_id),
            )
        return self.task_summary(task_id)

    def fail_task(self, task_id: str) -> None:
        with self._transaction() as connection:
            actual = self._task_actual(connection, task_id)
            connection.execute(
                """
                UPDATE task_budget
                SET actual_cny = ?, status = 'failed', updated_at = ?
                WHERE task_id = ?
                """,
                (str(actual), _now(), task_id),
            )

    def task_summary(self, task_id: str) -> dict[str, str]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM task_budget WHERE task_id = ?", (task_id,)
            ).fetchone()
            if not row:
                raise KeyError(task_id)
            return dict(row)

    def month_summary(self, month: str | None = None) -> dict[str, object]:
        month = month or _month()
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM task_budget WHERE month = ? ORDER BY created_at", (month,)
            ).fetchall()
            used = self._month_reserved_or_spent(connection, month)
        return {
            "month": month,
            "quality_trial_unlimited": self.settings.quality_trial_unlimited,
            "internal_used_or_reserved_cny": str(used),
            "warning_cny": str(self.settings.monthly_warning_cny),
            "hard_stop_cny": str(self.settings.monthly_hard_stop_cny),
            "external_cap_cny": str(self.settings.external_monthly_cap_cny),
            "deepseek_task_cap_cny": str(self.settings.deepseek_task_cap_cny),
            "kimi_task_cap_cny": str(self.settings.kimi_task_cap_cny),
            "other_provider_task_cap_cny": str(
                self.settings.other_provider_task_cap_cny
            ),
            "monthly_per_model_cap_cny": str(self.settings.monthly_per_model_cap_cny),
            "tasks": [dict(row) for row in rows],
        }

    def export_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.month_summary(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _month_reserved_or_spent(connection: sqlite3.Connection, month: str) -> Decimal:
        rows = connection.execute(
            "SELECT reserved_cny, actual_cny, status FROM task_budget WHERE month = ?", (month,)
        ).fetchall()
        return sum(
            (
                Decimal(row["actual_cny"])
                if row["status"] in {"completed", "failed"}
                else Decimal(row["reserved_cny"])
            )
            for row in rows
        )

    @staticmethod
    def _budget_group_calls_reserved_or_spent(
        connection: sqlite3.Connection, budget_group_id: str
    ) -> Decimal:
        rows = connection.execute(
            """
            SELECT model_call.estimated_cny, model_call.actual_cny
            FROM model_call
            JOIN task_budget ON task_budget.task_id = model_call.task_id
            WHERE task_budget.budget_group_id = ?
            """,
            (budget_group_id,),
        ).fetchall()
        return sum(
            Decimal(row["actual_cny"])
            if row["actual_cny"] is not None
            else Decimal(row["estimated_cny"])
            for row in rows
        )

    @staticmethod
    def _budget_group_provider_calls_reserved_or_spent(
        connection: sqlite3.Connection, budget_group_id: str, provider: str
    ) -> Decimal:
        rows = connection.execute(
            """
            SELECT model_call.estimated_cny, model_call.actual_cny
            FROM model_call
            JOIN task_budget ON task_budget.task_id = model_call.task_id
            WHERE task_budget.budget_group_id = ? AND model_call.provider = ?
            """,
            (budget_group_id, provider),
        ).fetchall()
        return sum(
            Decimal(row["actual_cny"])
            if row["actual_cny"] is not None
            else Decimal(row["estimated_cny"])
            for row in rows
        )

    @staticmethod
    def _month_provider_calls_reserved_or_spent(
        connection: sqlite3.Connection, month: str, provider: str
    ) -> Decimal:
        rows = connection.execute(
            """
            SELECT model_call.estimated_cny, model_call.actual_cny
            FROM model_call
            JOIN task_budget ON task_budget.task_id = model_call.task_id
            WHERE task_budget.month = ? AND model_call.provider = ?
            """,
            (month, provider),
        ).fetchall()
        return sum(
            Decimal(row["actual_cny"])
            if row["actual_cny"] is not None
            else Decimal(row["estimated_cny"])
            for row in rows
        )

    @staticmethod
    def _task_actual(connection: sqlite3.Connection, task_id: str) -> Decimal:
        rows = connection.execute(
            "SELECT estimated_cny, actual_cny, status FROM model_call WHERE task_id = ?",
            (task_id,),
        ).fetchall()
        return sum(
            Decimal(row["actual_cny"])
            if row["actual_cny"] is not None
            else Decimal(row["estimated_cny"])
            for row in rows
        )


def _month() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m")


def _now() -> str:
    return datetime.now(SHANGHAI).isoformat()


def _budget_group_id(task_id: str) -> str:
    for marker in (":gh-", ":local-"):
        if marker in task_id:
            return task_id.split(marker, 1)[0]
    return task_id
