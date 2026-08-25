from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from .schemas import PaperRecord, normalize_title


def deduplicate_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Merge records by permanent identifier, then normalized title and nearby year."""
    canonical: list[PaperRecord] = []
    id_index: dict[str, int] = {}
    title_index: dict[tuple[str, int], int] = {}
    fuzzy_buckets: dict[str, list[int]] = defaultdict(list)

    for paper in papers:
        identifiers = _identifier_keys(paper)
        match_index = next((id_index[key] for key in identifiers if key in id_index), None)
        title = normalize_title(paper.title)
        year = paper.year or 0
        if match_index is None and title:
            for candidate_year in {year - 1, year, year + 1, 0}:
                key = (title, candidate_year)
                if key in title_index:
                    match_index = title_index[key]
                    break
        if match_index is None and len(title) >= 24:
            bucket = title[:18]
            for candidate_index in fuzzy_buckets.get(bucket, []):
                candidate = canonical[candidate_index]
                if candidate.year and paper.year and abs(candidate.year - paper.year) > 1:
                    continue
                ratio = SequenceMatcher(
                    None, title, normalize_title(candidate.title), autojunk=False
                ).ratio()
                if ratio >= 0.96:
                    match_index = candidate_index
                    break

        if match_index is None:
            match_index = len(canonical)
            canonical.append(paper.model_copy(deep=True))
        else:
            canonical[match_index] = merge_papers(canonical[match_index], paper)

        merged = canonical[match_index]
        for identifier in _identifier_keys(merged):
            id_index[identifier] = match_index
        normalized = normalize_title(merged.title)
        merged_year = merged.year or 0
        if normalized:
            title_index[(normalized, merged_year)] = match_index
            fuzzy_buckets[normalized[:18]].append(match_index)

    return canonical


def merge_papers(left: PaperRecord, right: PaperRecord) -> PaperRecord:
    data = left.model_dump()
    for field in (
        "abstract",
        "publication_date",
        "venue",
        "work_type",
        "doi",
        "pmid",
        "pmcid",
        "arxiv_id",
        "landing_page_url",
        "open_access_pdf_url",
        "open_access_license",
    ):
        left_value = data.get(field)
        right_value = getattr(right, field)
        if not left_value or (
            field == "abstract"
            and right_value
            and len(str(right_value)) > len(str(left_value or ""))
        ):
            data[field] = right_value
    if not data.get("year") and right.year:
        data["year"] = right.year
    if len(right.title) > len(str(data.get("title") or "")):
        data["title"] = right.title
    data["authors"] = _unique([*left.authors, *right.authors])
    data["source_names"] = _unique([*left.source_names, *right.source_names])
    data["source_ids"] = {**left.source_ids, **right.source_ids}
    data["topics"] = _unique([*left.topics, *right.topics])[:20]
    data["quality_flags"] = _unique([*left.quality_flags, *right.quality_flags])
    data["referenced_work_ids"] = _unique(
        [*left.referenced_work_ids, *right.referenced_work_ids]
    )[:500]
    data["citation_count"] = max(left.citation_count, right.citation_count)
    data["influential_citation_count"] = max(
        left.influential_citation_count, right.influential_citation_count
    )
    data["source_relevance"] = max(left.source_relevance, right.source_relevance)
    return PaperRecord.model_validate(data)


def _identifier_keys(paper: PaperRecord) -> list[str]:
    keys: list[str] = []
    if paper.doi:
        keys.append(f"doi:{paper.doi}")
    if paper.pmid:
        keys.append(f"pmid:{paper.pmid}")
    if paper.arxiv_id:
        keys.append(f"arxiv:{paper.arxiv_id.lower()}")
    for source, identifier in paper.source_ids.items():
        if identifier:
            keys.append(f"{source}:{identifier}")
    return keys


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
