from __future__ import annotations

from typing import Any

from ..schemas import PaperRecord, TaskSpec
from .base import SearchSource, SourceError, first_text, safe_int


class OpenAlexSource(SearchSource):
    name = "openalex"
    endpoint = "https://api.openalex.org/works"

    async def search(self, query: str, task: TaskSpec, limit: int) -> list[PaperRecord]:
        raise SourceError(
            "OpenAlex is disabled in anonymous-only mode because its current API requires "
            "an account-linked key"
        )

    def _parse(self, item: dict[str, Any], rank: int) -> PaperRecord | None:
        title = first_text(item.get("title") or item.get("display_name"))
        work_id = first_text(item.get("id"))
        if not title or not work_id:
            return None
        doi = first_text(item.get("doi"))
        authors: list[str] = []
        for authorship in item.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            name = first_text(author.get("display_name")) if isinstance(author, dict) else None
            if name:
                authors.append(name)
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {} if isinstance(primary_location, dict) else {}
        best_oa = item.get("best_oa_location") or {}
        abstract = self._decode_abstract(item.get("abstract_inverted_index"))
        topics: list[str] = []
        for topic in item.get("topics") or []:
            if isinstance(topic, dict):
                topic_name = first_text(topic.get("display_name"))
                if topic_name:
                    topics.append(topic_name)
        return PaperRecord(
            record_id=f"openalex:{work_id.rsplit('/', 1)[-1]}",
            title=title,
            abstract=abstract,
            year=safe_int(item.get("publication_year")) or None,
            publication_date=first_text(item.get("publication_date")),
            authors=authors,
            venue=first_text(source.get("display_name")) if isinstance(source, dict) else None,
            work_type=first_text(item.get("type")),
            doi=doi,
            citation_count=safe_int(item.get("cited_by_count")),
            source_names=[self.name],
            source_ids={self.name: work_id},
            landing_page_url=first_text(primary_location.get("landing_page_url"))
            if isinstance(primary_location, dict)
            else None,
            open_access_pdf_url=first_text(best_oa.get("pdf_url"))
            if isinstance(best_oa, dict)
            else None,
            open_access_license=first_text(best_oa.get("license"))
            if isinstance(best_oa, dict)
            else None,
            topics=topics[:10],
            source_relevance=1.0 / (1.0 + rank),
            referenced_work_ids=[
                str(value).rsplit("/", 1)[-1]
                for value in item.get("referenced_works") or []
                if value
            ],
        )

    async def fetch_by_ids(
        self, ids: list[str], task: TaskSpec, limit: int
    ) -> list[PaperRecord]:
        raise SourceError(
            "OpenAlex citation expansion is disabled in anonymous-only mode"
        )

    async def citing_works(
        self, openalex_id: str, task: TaskSpec, limit: int
    ) -> list[PaperRecord]:
        raise SourceError(
            "OpenAlex citation expansion is disabled in anonymous-only mode"
        )

    @staticmethod
    def _decode_abstract(index: Any) -> str | None:
        if not isinstance(index, dict) or not index:
            return None
        positions: list[tuple[int, str]] = []
        for word, offsets in index.items():
            if not isinstance(word, str) or not isinstance(offsets, list):
                continue
            for offset in offsets:
                if isinstance(offset, int):
                    positions.append((offset, word))
        if not positions:
            return None
        positions.sort()
        return " ".join(word for _, word in positions)
