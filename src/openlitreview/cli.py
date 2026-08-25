from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .audit import audit_run
from .benchmark import run_benchmark
from .budget import BudgetLedger
from .config import dump_task_contract, load_task
from .pipeline import execute_pipeline, execution_id
from .schemas import BudgetSettings
from .search import run_search
from .storage import prepare_run_directory, write_search_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openlitreview",
        description="On-demand, auditable academic literature-review pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a task file")
    validate.add_argument("task")

    contract = subparsers.add_parser("contract", help="Write a human-readable task contract")
    contract.add_argument("task")
    contract.add_argument("--output", required=True)

    search = subparsers.add_parser("search", help="Run metadata search without model calls")
    search.add_argument("task")
    search.add_argument("--output", required=True)

    run = subparsers.add_parser("run", help="Run the configured review pipeline")
    run.add_argument("task")
    run.add_argument("--output", required=True)
    run.add_argument("--ledger", default="state/budget.sqlite")
    run.add_argument("--confirmed", action="store_true")

    reserve = subparsers.add_parser(
        "budget-reserve", help="Reserve the task maximum before model calls"
    )
    reserve.add_argument("task")
    reserve.add_argument("--ledger", default="state/budget.sqlite")

    budget = subparsers.add_parser("budget-summary", help="Show the current monthly budget")
    budget.add_argument("task")
    budget.add_argument("--ledger", default="state/budget.sqlite")

    benchmark_reserve = subparsers.add_parser(
        "benchmark-budget-reserve",
        help="Durably reserve the blind benchmark maximum before paid calls",
    )
    benchmark_reserve.add_argument("--ledger", default="state/budget.sqlite")

    benchmark = subparsers.add_parser("benchmark", help="Run a blinded multi-model benchmark")
    benchmark.add_argument("--suite", default="benchmarks/academic_zh_v1.json")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--ledger", default="state/budget.sqlite")
    benchmark.add_argument(
        "--models",
        nargs="+",
        default=["deepseek-v4-pro", "kimi-k2.6", "doubao-seed-2.1-pro"],
    )
    benchmark.add_argument(
        "--confirmed",
        action="store_true",
        help="Confirm the paid benchmark and its maximum reservation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            task = load_task(args.task)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "task_id": task.resolved_task_id(),
                        "queries": task.base_queries(),
                        "models_enabled": task.models.enabled,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "contract":
            task = load_task(args.task)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            dump_task_contract(task, output)
            print(str(output.resolve()))
            return 0
        if args.command == "search":
            task = load_task(args.task)
            output = prepare_run_directory(args.output)
            dump_task_contract(task, output / "task_contract.json")
            run = asyncio.run(run_search(task))
            write_search_outputs(run, task, output)
            audit_run(task, run, None, None, None, output)
            print(json.dumps({"output": str(output), "records": len(run.papers)}))
            return 0
        if args.command == "run":
            result = asyncio.run(
                execute_pipeline(
                    args.task,
                    args.output,
                    args.ledger,
                    human_confirmed=args.confirmed,
                )
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["quality"]["status"] == "pass" else 2
        if args.command == "budget-reserve":
            task = load_task(args.task)
            if not task.models.enabled:
                print(json.dumps({"status": "skipped", "reason": "models_disabled"}))
                return 0
            run_budget_id = execution_id(task.resolved_task_id())
            ledger = BudgetLedger(args.ledger, task.budget)
            reservation = ledger.reserve_task(run_budget_id)
            print(json.dumps(reservation, ensure_ascii=False, indent=2))
            return 0
        if args.command == "budget-summary":
            task = load_task(args.task)
            ledger = BudgetLedger(args.ledger, task.budget)
            print(json.dumps(ledger.month_summary(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "benchmark-budget-reserve":
            settings = BudgetSettings(task_reservation_cny=30, single_request_cap_cny=5)
            ledger = BudgetLedger(args.ledger, settings)
            benchmark_id = execution_id("phase0-model-benchmark")
            reservation = ledger.reserve_task(benchmark_id)
            print(json.dumps(reservation, ensure_ascii=False, indent=2))
            return 0
        if args.command == "benchmark":
            if not args.confirmed:
                raise ValueError("Paid model benchmark requires --confirmed")
            result = asyncio.run(
                run_benchmark(args.suite, args.output, args.ledger, args.models)
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
