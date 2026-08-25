from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from .citation_chase import expand_openalex_citations
from .dedupe import deduplicate_papers
from .ranking import filter_excluded, rank_papers
from .schemas import PaperRecord, SearchRun, SearchSourceName, TaskSpec
from .sources import (
    CrossrefSource,
    EuropePMCSource,
    OpenAlexSource,
    SearchSource,
    SemanticScholarSource,
)

SOURCE_CLASSES: dict[SearchSourceName, type[SearchSource]] = {
    SearchSourceName.OPENALEX: OpenAlexSource,
    SearchSourceName.CROSSREF: CrossrefSource,
    SearchSourceName.SEMANTIC_SCHOLAR: SemanticScholarSource,
    SearchSourceName.EUROPE_PMC: EuropePMCSource,
}


async def run_search(task: TaskSpec, queries: list[str] | None = None) -> SearchRun:
    queries = queries or task.base_queries()
    started_at = datetime.now(UTC).isoformat()
    jobs = [
        _search_source(source_name, queries, task)
        for source_name in task.search.sources
        if source_name in SOURCE_CLASSES
    ]
    results = await asyncio.gather(*jobs)
    all_papers: list[PaperRecord] = []
    source_status: dict[str, dict[str, Any]] = {}
    for source_name, papers, status in results:
        source_status[source_name] = status
        all_papers.extend(papers)
    raw_count = len(all_papers)
    deduplicated = deduplicate_papers(all_papers)
    filtered = filter_excluded(deduplicated, task)
    ranked = rank_papers(filtered, task)[: task.search.max_candidates]
    if SearchSourceName.OPENALEX in task.search.sources:
        ranked, citation_status = await expand_openalex_citations(ranked, task)
        source_status["citation_chase"] = citation_status
    return SearchRun(
        task_id=task.resolved_task_id(),
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        queries=queries,
        source_status=source_status,
        raw_record_count=raw_count,
        deduplicated_record_count=len(deduplicated),
        papers=ranked,
    )


async def _search_source(
    source_name: SearchSourceName, queries: list[str], task: TaskSpec
) -> tuple[str, list[PaperRecord], dict[str, Any]]:
    source = SOURCE_CLASSES[source_name]()
    papers: list[PaperRecord] = []
    query_log: list[dict[str, Any]] = []
    try:
        for query in queries[: task.search.max_queries_per_source]:
            try:
                records = await source.search(
                    query,
                    task,
                    limit=task.search.max_per_query_per_source,
                )
                papers.extend(records)
                query_log.append({"query": query, "records": len(records), "status": "ok"})
            except Exception as exc:  # source failures must not collapse other sources
                query_log.append(
                    {"query": query, "records": 0, "status": "failed", "error": str(exc)[:300]}
                )
        success_count = sum(item["status"] == "ok" for item in query_log)
        status = {
            "status": "ok" if success_count else "failed",
            "queries": query_log,
            "records": len(papers),
        }
        return source_name.value, papers, status
    finally:
        await source.close()
