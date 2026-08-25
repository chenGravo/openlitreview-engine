from __future__ import annotations

from typing import Any

from ..schemas import PaperRecord, TaskSpec
from .base import SearchSource, first_text, safe_int


class SemanticScholarSource(SearchSource):
    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    async def search(self, query: str, task: TaskSpec, limit: int) -> list[PaperRecord]:
        token: str | None = None
        papers: list[PaperRecord] = []
        while len(papers) < limit:
            params: dict[str, Any] = {
                "query": query,
                "fields": (
                    "title,abstract,year,authors,venue,citationCount,influentialCitationCount,"
                    "publicationDate,publicationTypes,externalIds,url,openAccessPdf,fieldsOfStudy"
                ),
                "limit": min(1000, limit - len(papers)),
            }
            if task.year_from or task.year_to:
                start = task.year_from or ""
                end = task.year_to or ""
                params["year"] = f"{start}-{end}"
            if token:
                params["token"] = token
            payload = await self.get_json(self.endpoint, params=params)
            data = payload.get("data") or []
            if not isinstance(data, list) or not data:
                break
            for rank, item in enumerate(data, start=len(papers)):
                if isinstance(item, dict):
                    parsed = self._parse(item, rank)
                    if parsed:
                        papers.append(parsed)
            next_token = payload.get("token")
            token = str(next_token) if next_token else None
            if not token:
                break
        return papers[:limit]

    def _parse(self, item: dict[str, Any], rank: int) -> PaperRecord | None:
        title = first_text(item.get("title"))
        paper_id = first_text(item.get("paperId"))
        if not title or not paper_id:
            return None
        external = item.get("externalIds") or {}
        if not isinstance(external, dict):
            external = {}
        authors = [
            name
            for author in item.get("authors") or []
            if isinstance(author, dict) and (name := first_text(author.get("name")))
        ]
        oa_pdf = item.get("openAccessPdf") or {}
        fields = item.get("fieldsOfStudy") or []
        topics = [str(field) for field in fields if field]
        publication_types = item.get("publicationTypes") or []
        return PaperRecord(
            record_id=f"semantic_scholar:{paper_id}",
            title=title,
            abstract=first_text(item.get("abstract")),
            year=safe_int(item.get("year")) or None,
            publication_date=first_text(item.get("publicationDate")),
            authors=authors,
            venue=first_text(item.get("venue")),
            work_type=first_text(publication_types),
            doi=first_text(external.get("DOI")),
            pmid=first_text(external.get("PubMed")),
            arxiv_id=first_text(external.get("ArXiv")),
            citation_count=safe_int(item.get("citationCount")),
            influential_citation_count=safe_int(item.get("influentialCitationCount")),
            source_names=[self.name],
            source_ids={self.name: paper_id},
            landing_page_url=first_text(item.get("url")),
            open_access_pdf_url=first_text(oa_pdf.get("url"))
            if isinstance(oa_pdf, dict)
            else None,
            # Semantic Scholar's `status` is an OA route, not a reuse license.
            open_access_license=None,
            topics=topics,
            source_relevance=1.0 / (1.0 + rank),
        )
