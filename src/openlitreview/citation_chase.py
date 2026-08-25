from __future__ import annotations

from typing import Any

from .dedupe import deduplicate_papers
from .ranking import rank_papers
from .schemas import PaperRecord, TaskSpec
from .sources.openalex import OpenAlexSource


async def expand_openalex_citations(
    papers: list[PaperRecord], task: TaskSpec
) -> tuple[list[PaperRecord], dict[str, Any]]:
    if task.search.citation_seed_count == 0 or task.search.max_citation_expansion == 0:
        return papers, {"status": "skipped", "saturation_reached": False, "rounds": []}
    source = OpenAlexSource()
    current = papers
    expanded_seed_ids: set[str] = set()
    total_added = 0
    rounds: list[dict[str, Any]] = []
    saturation_reached = False
    try:
        for round_number in range(1, task.search.saturation_rounds + 1):
            seeds = []
            for paper in current:
                openalex_id = paper.source_ids.get("openalex")
                if not openalex_id or openalex_id in expanded_seed_ids:
                    continue
                seeds.append(paper)
                expanded_seed_ids.add(openalex_id)
                if len(seeds) >= task.search.citation_seed_count:
                    break
            if not seeds:
                rounds.append(
                    {"round": round_number, "status": "no_new_seeds", "new_records": 0}
                )
                break
            remaining = max(0, task.search.max_citation_expansion - total_added)
            if remaining == 0:
                break
            backward_ids = list(
                dict.fromkeys(
                    reference
                    for seed in seeds
                    for reference in seed.referenced_work_ids[:100]
                )
            )
            backward_limit = min(remaining // 2 or remaining, len(backward_ids), 500)
            backward = await source.fetch_by_ids(backward_ids, task, backward_limit)
            forward: list[PaperRecord] = []
            forward_per_seed = min(50, max(10, remaining // max(len(seeds), 1)))
            for seed in seeds:
                if len(forward) + len(backward) >= remaining:
                    break
                openalex_id = seed.source_ids.get("openalex")
                if openalex_id:
                    forward.extend(
                        await source.citing_works(openalex_id, task, forward_per_seed)
                    )
            before = len(current)
            existing_keys = {paper.canonical_key() for paper in current}
            merged = deduplicate_papers([*current, *backward, *forward])
            new_count = max(0, len(merged) - before)
            total_added += new_count
            current = rank_papers(merged, task)[: task.search.max_candidates]
            screened = current[: task.search.screening_pool]
            new_relevant_count = sum(
                paper.canonical_key() not in existing_keys for paper in screened
            )
            ratio = new_relevant_count / max(min(before, task.search.screening_pool), 1)
            rounds.append(
                {
                    "round": round_number,
                    "status": "ok",
                    "seed_count": len(seeds),
                    "backward_records": len(backward),
                    "forward_records": len(forward),
                    "new_unique_records": new_count,
                    "new_relevant_records_in_screening_pool": new_relevant_count,
                    "new_relevant_ratio": round(ratio, 6),
                }
            )
            if ratio < task.search.saturation_new_relevant_ratio:
                saturation_reached = True
                break
        return current, {
            "status": "ok",
            "records_added": total_added,
            "saturation_reached": saturation_reached,
            "rounds": rounds,
        }
    except Exception as exc:
        return current, {
            "status": "failed",
            "records_added": total_added,
            "saturation_reached": False,
            "rounds": rounds,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    finally:
        await source.close()
