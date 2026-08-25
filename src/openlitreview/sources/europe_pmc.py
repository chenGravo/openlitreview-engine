from __future__ import annotations

from typing import Any

from ..schemas import PaperRecord, TaskSpec
from .base import SearchSource, first_text, safe_int


class EuropePMCSource(SearchSource):
    name = "europe_pmc"
    endpoint = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    async def search(self, query: str, task: TaskSpec, limit: int) -> list[PaperRecord]:
        clauses = [f"({query})"]
        if task.year_from and task.year_to:
            clauses.append(f"FIRST_PDATE:[{task.year_from}-01-01 TO {task.year_to}-12-31]")
        elif task.year_from:
            clauses.append(f"FIRST_PDATE:[{task.year_from}-01-01 TO 2100-12-31]")
        elif task.year_to:
            clauses.append(f"FIRST_PDATE:[1800-01-01 TO {task.year_to}-12-31]")
        cursor = "*"
        papers: list[PaperRecord] = []
        while len(papers) < limit:
            page_size = min(1000, limit - len(papers))
            params: dict[str, Any] = {
                "query": " AND ".join(clauses),
                "format": "json",
                "resultType": "core",
                "pageSize": page_size,
                "cursorMark": cursor,
            }
            payload = await self.get_json(self.endpoint, params=params)
            result_list = payload.get("resultList") or {}
            results = result_list.get("result") or [] if isinstance(result_list, dict) else []
            if not isinstance(results, list) or not results:
                break
            for rank, item in enumerate(results, start=len(papers)):
                if isinstance(item, dict):
                    parsed = self._parse(item, rank)
                    if parsed:
                        papers.append(parsed)
            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor or len(results) < page_size:
                break
            cursor = str(next_cursor)
        return papers[:limit]

    def _parse(self, item: dict[str, Any], rank: int) -> PaperRecord | None:
        title = first_text(item.get("title"))
        source_id = first_text(item.get("id"))
        if not title or not source_id:
            return None
        author_list = item.get("authorList") or {}
        author_items = author_list.get("author") or [] if isinstance(author_list, dict) else []
        authors: list[str] = []
        for author in author_items:
            if isinstance(author, dict):
                name = first_text(author.get("fullName"))
                if name:
                    authors.append(name)
        full_text_urls = item.get("fullTextUrlList") or {}
        url_items = (
            full_text_urls.get("fullTextUrl") or []
            if isinstance(full_text_urls, dict)
            else []
        )
        pdf_url = None
        license_name = first_text(item.get("license"))
        for url_item in url_items:
            if not isinstance(url_item, dict):
                continue
            document_style = str(url_item.get("documentStyle") or "").lower()
            availability = str(url_item.get("availability") or "").lower()
            if document_style == "pdf" and availability in {"open access", "free"}:
                pdf_url = first_text(url_item.get("url"))
                if pdf_url:
                    break
        pmcid = first_text(item.get("pmcid"))
        landing = f"https://europepmc.org/article/{item.get('source', 'MED')}/{source_id}"
        return PaperRecord(
            record_id=f"europe_pmc:{item.get('source', 'MED')}:{source_id}",
            title=title,
            abstract=first_text(item.get("abstractText")),
            year=safe_int(item.get("pubYear")) or None,
            publication_date=first_text(item.get("firstPublicationDate")),
            authors=authors,
            venue=first_text(item.get("journalTitle")),
            work_type=first_text(item.get("pubType")),
            doi=first_text(item.get("doi")),
            pmid=first_text(item.get("pmid")),
            pmcid=pmcid,
            citation_count=safe_int(item.get("citedByCount")),
            source_names=[self.name],
            source_ids={self.name: source_id},
            landing_page_url=landing,
            open_access_pdf_url=pdf_url,
            open_access_license=license_name,
            topics=[str(term) for term in item.get("meshHeadingList", {}).get("meshHeading", [])]
            if isinstance(item.get("meshHeadingList"), dict)
            else [],
            source_relevance=1.0 / (1.0 + rank),
        )
