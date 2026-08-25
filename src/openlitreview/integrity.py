from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from .network import ANONYMOUS_JSON_HEADERS
from .schemas import PaperRecord

ADVERSE_UPDATE_TERMS = {
    "retraction": "retracted",
    "withdrawal": "withdrawn",
    "expression-of-concern": "expression_of_concern",
}


async def check_publication_updates(
    papers: list[PaperRecord], max_concurrency: int = 3
) -> list[PaperRecord]:
    semaphore = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient(
        headers=ANONYMOUS_JSON_HEADERS,
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
    ) as client:
        tasks = [_check_one(client, semaphore, paper) for paper in papers]
        return await asyncio.gather(*tasks)


async def _check_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    paper: PaperRecord,
) -> PaperRecord:
    copy = paper.model_copy(deep=True)
    if not paper.doi:
        copy.publication_status = "identifier_unavailable"
        copy.quality_flags.append("publication_update_check_limited")
        return copy
    try:
        async with semaphore:
            response = await client.get(
                f"https://api.crossref.org/works/{quote(paper.doi, safe='')}"
            )
            response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        updates = _extract_updates(message if isinstance(message, dict) else {})
        copy.publication_updates = updates
        adverse = next(
            (
                ADVERSE_UPDATE_TERMS[update["type"]]
                for update in updates
                if update.get("type") in ADVERSE_UPDATE_TERMS
            ),
            None,
        )
        copy.publication_status = adverse or ("updated" if updates else "no_adverse_update_found")
        if adverse:
            copy.quality_flags.append(f"publication_status_{adverse}")
        return copy
    except Exception:
        copy.publication_status = "check_failed"
        copy.quality_flags.append("publication_update_check_failed")
        return copy


def _extract_updates(message: dict[str, Any]) -> list[dict[str, str]]:
    updates: list[dict[str, str]] = []
    for item in message.get("update-to") or []:
        if not isinstance(item, dict):
            continue
        update_type = str(item.get("type") or "update").lower()
        updates.append(
            {
                "type": update_type,
                "doi": str(item.get("DOI") or ""),
                "label": str(item.get("label") or ""),
            }
        )
    relation = message.get("relation") or {}
    if isinstance(relation, dict):
        mapping = {
            "is-retracted-by": "retraction",
            "is-corrected-by": "correction",
            "is-updated-by": "update",
        }
        for key, update_type in mapping.items():
            for item in relation.get(key) or []:
                if isinstance(item, dict):
                    updates.append(
                        {
                            "type": update_type,
                            "doi": str(item.get("id") or ""),
                            "label": key,
                        }
                    )
    return updates
