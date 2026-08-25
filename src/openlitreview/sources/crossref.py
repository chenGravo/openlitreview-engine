from __future__ import annotations

import html
import re
from typing import Any

from ..schemas import PaperRecord, TaskSpec
from .base import SearchSource, first_text, safe_int


class CrossrefSource(SearchSource):
    name = "crossref"
    endpoint = "https://api.crossref.org/works"

    async def search(self, query: str, task: TaskSpec, limit: int) -> list[PaperRecord]:
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": min(limit, 1000),
            "select": (
                "DOI,title,abstract,author,published,published-online,published-print,"
                "container-title,type,is-referenced-by-count,URL,link,license,score"
            ),
        }
        filters: list[str] = []
        if task.year_from:
            filters.append(f"from-pub-date:{task.year_from}-01-01")
        if task.year_to:
            filters.append(f"until-pub-date:{task.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        payload = await self.get_json(self.endpoint, params=params)
        message = payload.get("message") or {}
        items = message.get("items") or [] if isinstance(message, dict) else []
        papers: list[PaperRecord] = []
        for rank, item in enumerate(items):
            if isinstance(item, dict):
                parsed = self._parse(item, rank)
                if parsed:
                    papers.append(parsed)
        return papers[:limit]

    def _parse(self, item: dict[str, Any], rank: int) -> PaperRecord | None:
        title = first_text(item.get("title"))
        doi = first_text(item.get("DOI"))
        if not title:
            return None
        authors: list[str] = []
        for author in item.get("author") or []:
            if not isinstance(author, dict):
                continue
            family = first_text(author.get("family")) or ""
            given = first_text(author.get("given")) or ""
            name = " ".join(part for part in (given, family) if part).strip()
            if name:
                authors.append(name)
        publication_date, year = self._published_date(item)
        abstract = first_text(item.get("abstract"))
        if abstract:
            abstract = html.unescape(re.sub(r"<[^>]+>", " ", abstract))
            abstract = re.sub(r"\s+", " ", abstract).strip()
        pdf_url = None
        for link in item.get("link") or []:
            if not isinstance(link, dict):
                continue
            content_type = str(link.get("content-type") or "").lower()
            if "pdf" in content_type:
                pdf_url = first_text(link.get("URL"))
                if pdf_url:
                    break
        licenses = item.get("license") or []
        license_url = None
        if licenses and isinstance(licenses[0], dict):
            license_url = first_text(licenses[0].get("URL"))
        record_suffix = doi or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
        return PaperRecord(
            record_id=f"crossref:{record_suffix}",
            title=title,
            abstract=abstract,
            year=year,
            publication_date=publication_date,
            authors=authors,
            venue=first_text(item.get("container-title")),
            work_type=first_text(item.get("type")),
            doi=doi,
            citation_count=safe_int(item.get("is-referenced-by-count")),
            source_names=[self.name],
            source_ids={self.name: doi or record_suffix},
            landing_page_url=first_text(item.get("URL")),
            open_access_pdf_url=pdf_url,
            open_access_license=license_url,
            source_relevance=1.0 / (1.0 + rank),
        )

    @staticmethod
    def _published_date(item: dict[str, Any]) -> tuple[str | None, int | None]:
        for key in ("published", "published-print", "published-online"):
            value = item.get(key)
            if not isinstance(value, dict):
                continue
            parts = value.get("date-parts")
            if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
                continue
            numbers = [safe_int(part) for part in parts[0] if safe_int(part)]
            if not numbers:
                continue
            year = numbers[0]
            publication_date = "-".join(f"{part:02d}" for part in numbers)
            return publication_date, year
        return None, None
