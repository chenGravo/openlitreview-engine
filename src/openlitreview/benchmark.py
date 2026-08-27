from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .budget import BudgetLedger
from .llm import LLMClient
from .pipeline import execution_id
from .schemas import BudgetSettings

BENCHMARK_SYSTEM = """You are completing a blinded Chinese academic synthesis test.
Use only the supplied synthetic evidence. Return strict JSON with keys `text` and `citations`.
The text must be concise Chinese academic prose. Citations must be selected only from supplied
evidence IDs. Do not invent papers, identifiers, data, mechanisms, or certainty."""


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    instruction: str
    evidence: list[dict[str, Any]]
    allowed_citations: set[str]
    required_citations: set[str]
    forbidden_patterns: tuple[str, ...]


async def run_benchmark(
    suite_path: str | Path,
    output_path: str | Path,
    ledger_path: str | Path,
    model_aliases: list[str],
) -> dict[str, Any]:
    cases = load_suite(suite_path)
    if len(model_aliases) < 2:
        raise ValueError("Blind benchmark requires at least two model aliases")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(
        ledger_path,
        BudgetSettings(
            task_reservation_cny=30,
            single_request_cap_cny=5,
            per_model_task_cap_cny=10,
            monthly_per_model_cap_cny=10,
        ),
    )
    run_id = execution_id("phase0-model-benchmark")
    ledger.reserve_task(run_id, Decimal("30"))
    client = LLMClient(ledger, run_id)
    labels = [f"candidate_{chr(65 + index)}" for index in range(len(model_aliases))]
    shuffled_models = list(model_aliases)
    random.SystemRandom().shuffle(shuffled_models)
    private_map = dict(zip(labels, shuffled_models, strict=True))
    blinded_results: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    try:
        for label, model_alias in private_map.items():
            for case in cases:
                prompt = json.dumps(
                    {
                        "case_id": case.case_id,
                        "instruction": case.instruction,
                        "evidence": case.evidence,
                        "required_schema": {"text": "", "citations": []},
                    },
                    ensure_ascii=False,
                )
                try:
                    response = await client.complete_json(
                        model_alias=model_alias,
                        system=BENCHMARK_SYSTEM,
                        prompt=prompt,
                        max_output_tokens=1_200,
                        temperature=0.0,
                    )
                    score = score_case(case, response)
                    blinded_results[label].append(
                        {
                            "case_id": case.case_id,
                            "category": case.category,
                            "response": response,
                            "automatic_score": score,
                        }
                    )
                except Exception as exc:
                    blinded_results[label].append(
                        {
                            "case_id": case.case_id,
                            "category": case.category,
                            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                            "automatic_score": {"total": 0, "failed": True},
                        }
                    )
        summary = summarize(blinded_results)
        (output / "blinded_results.json").write_text(
            json.dumps(blinded_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "score_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "human_scoring_sheet.md").write_text(
            render_human_sheet(blinded_results), encoding="utf-8"
        )
        private_dir = output / "private_work"
        private_dir.mkdir(exist_ok=True)
        (private_dir / "private_model_map.json").write_text(
            json.dumps(private_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        ledger.complete_task(run_id)
        ledger.export_json(private_dir / "budget_month.json")
        return summary
    except Exception:
        ledger.fail_task(run_id)
        raise


def load_suite(path: str | Path) -> list[BenchmarkCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Benchmark suite must be a non-empty JSON list")
    cases: list[BenchmarkCase] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each benchmark case must be an object")
        evidence = item.get("evidence") or []
        allowed = {
            str(value["id"])
            for value in evidence
            if isinstance(value, dict) and value.get("id")
        }
        cases.append(
            BenchmarkCase(
                case_id=str(item["case_id"]),
                category=str(item["category"]),
                instruction=str(item["instruction"]),
                evidence=evidence,
                allowed_citations=allowed,
                required_citations=set(item.get("required_citations") or []),
                forbidden_patterns=tuple(item.get("forbidden_patterns") or []),
            )
        )
    return cases


def score_case(case: BenchmarkCase, response: dict[str, Any]) -> dict[str, Any]:
    text = str(response.get("text") or "").strip()
    citations = {str(value) for value in response.get("citations") or []}
    invalid = citations - case.allowed_citations
    missing = case.required_citations - citations
    forbidden_hits = [
        pattern for pattern in case.forbidden_patterns if re.search(pattern, text, re.IGNORECASE)
    ]
    scores = {
        "valid_citations": 35 if not invalid else 0,
        "required_evidence": 25 if not missing else max(0, 25 - 10 * len(missing)),
        "no_forbidden_overclaim": 25 if not forbidden_hits else 0,
        "structured_and_nonempty": (
            15 if text and isinstance(response.get("citations"), list) else 0
        ),
    }
    return {
        "total": sum(scores.values()),
        "components": scores,
        "invalid_citations": sorted(invalid),
        "missing_required_citations": sorted(missing),
        "forbidden_hits": forbidden_hits,
    }


def summarize(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for label, rows in results.items():
        scores = [int(row["automatic_score"].get("total", 0)) for row in rows]
        candidates[label] = {
            "cases": len(rows),
            "automatic_average": round(sum(scores) / max(len(scores), 1), 2),
            "failed_cases": sum(bool(row.get("error")) for row in rows),
            "human_score_pending": True,
        }
    return {
        "candidates": candidates,
        "selection_status": "human_blind_scoring_required",
        "automatic_scores_are_not_sufficient_for_model_selection": True,
    }


def render_human_sheet(results: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# 三模型盲测人工评分表",
        "",
        "请在不知道模型身份的情况下按每题 100 分评分：证据忠实度35、中文综合25、"
        "结构遵循15、长文本取证15、稳定性5、成本5。",
        "",
    ]
    for label, rows in results.items():
        lines.extend(
            [
                f"## {label}",
                "",
                "| 题目 | 类别 | 自动分 | 人工分 | 备注 |",
                "|---|---|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['case_id']} | {row['category']} | "
                f"{row['automatic_score'].get('total', 0)} |  |  |"
            )
        for row in rows:
            response = row.get("response") or {}
            lines.extend(
                [
                    "",
                    f"### {row['case_id']} 回答",
                    "",
                    str(response.get("text") or f"运行失败：{row.get('error', '未知错误')}"),
                    "",
                    "证据编号："
                    + "、".join(str(value) for value in response.get("citations") or []),
                    "",
                    "人工评分：____ / 100；备注：________________",
                ]
            )
        lines.append("")
    return "\n".join(lines)
