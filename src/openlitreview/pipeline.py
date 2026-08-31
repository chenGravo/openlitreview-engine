from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .audit import audit_run
from .budget import BudgetLedger
from .config import dump_task_contract, load_task
from .evidence import extract_evidence_cards, load_evidence_seed
from .fulltext import collect_fulltexts
from .integrity import check_publication_updates
from .llm import LLMClient
from .prompts import QUERY_SYSTEM, query_expansion_prompt
from .render import render_documents
from .schemas import PaperRecord, TaskSpec
from .search import run_search
from .storage import prepare_run_directory, write_search_outputs
from .writer import generate_review


async def execute_pipeline(
    task_path: str | Path,
    output_path: str | Path,
    ledger_path: str | Path,
    *,
    human_confirmed: bool,
) -> dict[str, Any]:
    task = load_task(task_path)
    output = prepare_run_directory(output_path)
    dump_task_contract(task, output / "task_contract.json")
    ledger = BudgetLedger(ledger_path, task.budget)
    task_id = task.resolved_task_id()
    execution_id_for_run = execution_id(task_id)
    reserved = False
    try:
        queries = task.base_queries()
        client: LLMClient | None = None
        if task.models.enabled:
            if not human_confirmed:
                raise ValueError("Model-enabled runs require explicit human confirmation")
            ledger.reserve_task(
                execution_id_for_run, Decimal(str(task.budget.task_reservation_cny))
            )
            reserved = True
            client = LLMClient(ledger, execution_id_for_run)
            if not task.evidence_seed_file:
                queries = await _expand_queries(task, client)

        search_run = await run_search(task, queries)
        checked_count = min(len(search_run.papers), max(task.search.target_fulltexts * 2, 100))
        checked = await check_publication_updates(search_run.papers[:checked_count])
        search_run.papers = [*checked, *search_run.papers[checked_count:]]
        write_search_outputs(search_run, task, output)

        if not task.models.enabled or client is None:
            quality = audit_run(task, search_run, None, None, None, output)
            _write_provenance(
                output,
                task,
                queries,
                execution_id=execution_id_for_run,
                model_calls_enabled=False,
            )
            return {
                "task_id": task_id,
                "stage": "search_complete",
                "output": str(output),
                "quality": quality,
            }

        seed_cards, seed_log, seed_papers, seed_digest = load_evidence_seed(
            task_path, task.evidence_seed_file
        )
        selected_seed_papers = _select_seed_papers(
            seed_papers, seed_cards, task.search.target_fulltexts
        )
        search_run.papers = _merge_papers(selected_seed_papers, search_run.papers)
        write_search_outputs(search_run, task, output)
        seeded_keys = {paper.canonical_key() for paper in selected_seed_papers}
        remaining_target = max(task.search.target_fulltexts - len(selected_seed_papers), 0)
        fulltext_candidates = [
            paper for paper in search_run.papers if paper.canonical_key() not in seeded_keys
        ]
        fulltext_results = await collect_fulltexts(
            fulltext_candidates[: task.search.screening_pool],
            output / "private_work" / "fulltext",
            target=remaining_target,
        )
        new_evidence_papers = _select_evidence_papers(
            fulltext_candidates, fulltext_results, remaining_target
        )
        evidence_papers = _merge_papers(
            selected_seed_papers,
            new_evidence_papers,
            limit=task.search.target_fulltexts,
        )
        cards, extraction_log = await extract_evidence_cards(
            task,
            evidence_papers,
            fulltext_results,
            client,
            output,
            initial_cards=seed_cards,
            initial_log=seed_log,
        )
        evidence_paper_count = len({card.record_id for card in cards})
        if evidence_paper_count < task.quality.minimum_evidence_papers:
            quality = audit_run(task, search_run, cards, None, None, output)
            _write_provenance(
                output,
                task,
                queries,
                execution_id=execution_id_for_run,
                model_calls_enabled=True,
            )
            ledger.complete_task(execution_id_for_run)
            ledger.export_json(output / "private_work" / "budget_month.json")
            return {
                "task_id": task_id,
                "stage": "evidence_quality_gate",
                "output": str(output),
                "quality": quality,
                "evidence_extractions": len(extraction_log),
            }
        if task.compliance.ai_body_generation_allowed is False:
            quality = audit_run(task, search_run, cards, None, None, output)
            _write_provenance(
                output,
                task,
                queries,
                execution_id=execution_id_for_run,
                model_calls_enabled=True,
            )
            ledger.complete_task(execution_id_for_run)
            ledger.export_json(output / "private_work" / "budget_month.json")
            return {
                "task_id": task_id,
                "stage": "evidence_only_policy_gate",
                "output": str(output),
                "quality": quality,
                "evidence_extractions": len(extraction_log),
            }

        markdown, _, reviewer = await generate_review(
            task,
            evidence_papers,
            cards,
            client,
            output,
            initial_evidence_digest=seed_digest,
        )
        quality = audit_run(task, search_run, cards, markdown, reviewer, output)
        render_report = None
        if quality["status"] == "pass":
            render_report = render_documents(output)
        _write_provenance(
            output,
            task,
            queries,
            execution_id=execution_id_for_run,
            model_calls_enabled=True,
        )
        ledger.complete_task(execution_id_for_run)
        ledger.export_json(output / "private_work" / "budget_month.json")
        return {
            "task_id": task_id,
            "stage": "review_complete" if quality["status"] == "pass" else "review_blocked",
            "output": str(output),
            "quality": quality,
            "render": render_report,
        }
    except Exception:
        if reserved:
            ledger.fail_task(execution_id_for_run)
            ledger.export_json(output / "private_work" / "budget_month.json")
        raise


async def _expand_queries(task: TaskSpec, client: LLMClient) -> list[str]:
    payload = await client.complete_json(
        model_alias=task.models.cheap_model,
        system=QUERY_SYSTEM,
        prompt=query_expansion_prompt(task),
        max_output_tokens=2_000,
        temperature=0.1,
    )
    generated: list[str] = []
    for item in payload.get("queries") or []:
        if isinstance(item, dict):
            query = str(item.get("query") or "").strip()
        else:
            query = str(item).strip()
        if query and len(query) <= 500:
            generated.append(query)
    combined = list(dict.fromkeys([*task.base_queries(), *generated]))
    return combined[: task.search.max_queries_per_source]


def _select_evidence_papers(
    papers: list[PaperRecord], fulltexts: list[Any], target: int
) -> list[PaperRecord]:
    fulltext_ids = {
        result.record_id for result in fulltexts if getattr(result, "status", None) == "extracted"
    }
    fulltext_papers = [paper for paper in papers if paper.record_id in fulltext_ids]
    other_papers = [paper for paper in papers if paper.record_id not in fulltext_ids]
    return [*fulltext_papers, *other_papers][:target]


def _select_seed_papers(
    papers: list[PaperRecord], cards: list[Any], target: int
) -> list[PaperRecord]:
    card_ids = {card.record_id for card in cards}
    return [paper for paper in papers if paper.record_id in card_ids][:target]


def _merge_papers(
    preferred: list[PaperRecord],
    remaining: list[PaperRecord],
    *,
    limit: int | None = None,
) -> list[PaperRecord]:
    merged: list[PaperRecord] = []
    seen: set[str] = set()
    for paper in [*preferred, *remaining]:
        key = paper.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        merged.append(paper)
        if limit is not None and len(merged) >= limit:
            break
    return merged


def _write_provenance(
    output: Path,
    task: TaskSpec,
    queries: list[str],
    *,
    execution_id: str,
    model_calls_enabled: bool,
) -> None:
    payload = {
        "task_id": task.resolved_task_id(),
        "execution_id": execution_id,
        "engine_version": "0.1.0",
        "model_calls_enabled": model_calls_enabled,
        "models": task.models.model_dump(mode="json") if model_calls_enabled else None,
        "query_count": len(queries),
        "private_audit_only": True,
        "included_in_default_submission_package": False,
    }
    (output / "private_work" / "private_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def execution_id(task_id: str) -> str:
    github_run = os.getenv("GITHUB_RUN_ID")
    github_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    if github_run:
        return f"{task_id}:gh-{github_run}-{github_attempt}"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{task_id}:local-{timestamp}"
