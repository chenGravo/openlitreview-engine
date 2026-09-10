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


class _RequestPacer:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = self._next_start - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start = loop.time() + self.interval_seconds


async def check_publication_updates(
    papers: list[PaperRecord],
    max_concurrency: int = 3,
    *,
    request_interval_seconds: float = 0.15,
    max_attempts: int = 3,
) -> list[PaperRecord]:
    semaphore = asyncio.Semaphore(max_concurrency)
    pacer = _RequestPacer(request_interval_seconds)
    async with httpx.AsyncClient(
        headers=ANONYMOUS_JSON_HEADERS,
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
    ) as client:
        tasks = [
            _check_one(client, semaphore, pacer, paper, max_attempts=max_attempts)
            for paper in papers
        ]
        return await asyncio.gather(*tasks)


async def _check_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    pacer: _RequestPacer,
    paper: PaperRecord,
    *,
    max_attempts: int,
) -> PaperRecord:
    copy = paper.model_copy(deep=True)
    if not paper.doi:
        copy.publication_status = "identifier_unavailable"
        copy.quality_flags.append("publication_update_check_limited")
        return copy
    try:
        response = await _get_with_retry(
            client,
            semaphore,
            pacer,
            f"https://api.crossref.org/works/{quote(paper.doi, safe='')}",
            max_attempts=max_attempts,
        )
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


async def _get_with_retry(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    pacer: _RequestPacer,
    url: str,
    *,
    max_attempts: int,
) -> httpx.Response:
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            async with semaphore:
                await pacer.wait()
                response = await client.get(url)
                response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if attempt + 1 >= attempts or not _is_retryable(exc):
                raise
            await asyncio.sleep(_retry_delay_seconds(exc, attempt))
    raise RuntimeError("unreachable retry state")


def _is_retryable(exc: httpx.RequestError | httpx.HTTPStatusError) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    return exc.response.status_code == 429 or exc.response.status_code >= 500


def _retry_delay_seconds(
    exc: httpx.RequestError | httpx.HTTPStatusError,
    attempt: int,
) -> float:
    if isinstance(exc, httpx.HTTPStatusError):
        value = exc.response.headers.get("retry-after")
        if value:
            try:
                return min(max(float(value), 0.0), 5.0)
            except ValueError:
                pass
    return min(0.5 * (2**attempt), 5.0)


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
