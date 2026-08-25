from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..network import ANONYMOUS_JSON_HEADERS
from ..schemas import PaperRecord, TaskSpec


class SourceError(RuntimeError):
    """A recoverable scholarly-source failure."""


class SearchSource(ABC):
    name: str

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            headers=ANONYMOUS_JSON_HEADERS,
            timeout=httpx.Timeout(45.0, connect=15.0),
            follow_redirects=True,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        attempts: int = 4,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.get(url, params=params, headers=headers)
                if response.status_code == 429:
                    retry_after = min(float(response.headers.get("retry-after", "2")), 20.0)
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise SourceError(f"{self.name} returned non-object JSON")
                return payload
            except (httpx.HTTPError, ValueError, SourceError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2**attempt, 8))
        raise SourceError(f"{self.name} request failed after {attempts} attempts") from last_error

    @abstractmethod
    async def search(self, query: str, task: TaskSpec, limit: int) -> list[PaperRecord]:
        raise NotImplementedError


def first_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        for item in value:
            text = first_text(item)
            if text:
                return text
    return None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
