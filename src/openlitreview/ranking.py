from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date

from .schemas import PaperRecord, TaskSpec

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
PHRASE_SEPARATOR_RE = re.compile(r"[\W_]+", re.UNICODE)


def rank_papers(papers: list[PaperRecord], task: TaskSpec) -> list[PaperRecord]:
    papers = [paper for paper in papers if paper_within_year_range(paper, task)]
    if not papers:
        return []
    scope_terms = _scope_terms(task)
    query_text = " ".join(scope_terms) or task.research_question
    query_tokens = tokenize(query_text)
    query_counts = Counter(query_tokens)
    documents = [tokenize(_document_text(paper)) for paper in papers]
    document_counts = [Counter(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(len(document) for document in documents) / max(len(documents), 1)

    bm25_values: list[float] = []
    title_values: list[float] = []
    scope_values: list[float] = []
    primary_scope_values: list[float] = []
    impact_values: list[float] = []
    foundational_impact_values: list[float] = []
    recency_values: list[float] = []
    completeness_values: list[float] = []
    source_values: list[float] = []
    current_year = date.today().year

    for paper, document, counts in zip(papers, documents, document_counts, strict=True):
        bm25_values.append(
            _bm25(
                query_counts,
                counts,
                len(document),
                average_length,
                document_frequency,
                len(documents),
            )
        )
        title_tokens = set(tokenize(paper.title))
        query_set = set(query_tokens)
        title_values.append(len(title_tokens & query_set) / max(len(query_set), 1))
        document_phrase = _normalize_phrase(_document_text(paper))
        matched_terms = [term for term in scope_terms if _phrase_occurs(term, document_phrase)]
        scope_values.append(len(matched_terms) / max(len(scope_terms), 1))
        primary_scope_values.append(
            1.0
            if scope_terms and _phrase_occurs(scope_terms[0], document_phrase)
            else 0.0
        )
        age = max(1, current_year - (paper.year or current_year) + 1)
        impact_values.append(math.log1p(paper.citation_count) / (age**0.35))
        foundational_impact_values.append(math.log1p(paper.citation_count))
        recency_values.append(math.exp(-max(0, age - 1) / 12))
        completeness = sum(
            (
                bool(paper.abstract),
                bool(paper.doi or paper.pmid or paper.arxiv_id),
                bool(paper.authors),
                bool(paper.venue),
                bool(paper.open_access_pdf_url),
            )
        ) / 5
        completeness_values.append(completeness)
        source_values.append(min(len(paper.source_names), 4) / 4)

    normalized_components = {
        "relevance": _normalize(bm25_values),
        "title_match": _normalize(title_values),
        "scope_match": scope_values,
        "primary_scope_match": primary_scope_values,
        "impact": _normalize(impact_values),
        "foundational_impact": _normalize(foundational_impact_values),
        "recency": _normalize(recency_values),
        "evidence_availability": completeness_values,
        "source_confirmation": source_values,
    }
    weights = {
        "relevance": 0.30,
        "title_match": 0.10,
        "scope_match": 0.15,
        "primary_scope_match": 0.15,
        "impact": 0.08,
        "foundational_impact": 0.10,
        "recency": 0.03,
        "evidence_availability": 0.06,
        "source_confirmation": 0.03,
    }
    ranked: list[PaperRecord] = []
    for index, paper in enumerate(papers):
        breakdown = {
            component: round(values[index], 6)
            for component, values in normalized_components.items()
        }
        score = sum(weights[key] * breakdown[key] for key in weights)
        copy = paper.model_copy(deep=True)
        copy.rank_score = round(score, 6)
        copy.rank_breakdown = breakdown
        if not copy.abstract:
            _append_flag(copy, "abstract_missing")
        if not (copy.doi or copy.pmid or copy.arxiv_id):
            _append_flag(copy, "persistent_identifier_missing")
        if not copy.open_access_pdf_url:
            _append_flag(copy, "open_fulltext_unconfirmed")
        if scope_terms and breakdown["scope_match"] == 0:
            _append_flag(copy, "topical_anchor_missing")
        elif scope_terms and breakdown["primary_scope_match"] == 0:
            _append_flag(copy, "primary_scope_missing")
        ranked.append(copy)
    ranked.sort(key=lambda paper: (paper.rank_score, paper.citation_count), reverse=True)
    return ranked


def filter_excluded(papers: list[PaperRecord], task: TaskSpec) -> list[PaperRecord]:
    if not task.exclude_terms:
        return papers
    excluded = [term.lower() for term in task.exclude_terms]
    return [
        paper
        for paper in papers
        if not any(term in f"{paper.title} {paper.abstract or ''}".lower() for term in excluded)
    ]


def paper_within_year_range(paper: PaperRecord, task: TaskSpec) -> bool:
    """Enforce configured dates locally instead of trusting remote source filters."""
    if task.year_from is None and task.year_to is None:
        return True
    if paper.year is None:
        return False
    if task.year_from is not None and paper.year < task.year_from:
        return False
    return task.year_to is None or paper.year <= task.year_to


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _document_text(paper: PaperRecord) -> str:
    return " ".join([paper.title, paper.abstract or "", " ".join(paper.topics)])


def _scope_terms(task: TaskSpec) -> list[str]:
    terms = [
        normalized
        for value in [*task.keywords, *task.include_terms]
        if (normalized := _normalize_phrase(value))
    ]
    return list(dict.fromkeys(terms))


def _normalize_phrase(text: str) -> str:
    return PHRASE_SEPARATOR_RE.sub(" ", text.casefold()).strip()


def _phrase_occurs(term: str, normalized_document: str) -> bool:
    return f" {term} " in f" {normalized_document} "


def _append_flag(paper: PaperRecord, flag: str) -> None:
    if flag not in paper.quality_flags:
        paper.quality_flags.append(flag)


def _bm25(
    query: Counter[str],
    document: Counter[str],
    document_length: int,
    average_length: float,
    document_frequency: Counter[str],
    document_count: int,
) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    for token, query_frequency in query.items():
        frequency = document.get(token, 0)
        if frequency == 0:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        denominator = frequency + k1 * (
            1 - b + b * document_length / max(average_length, 1)
        )
        score += query_frequency * idf * frequency * (k1 + 1) / denominator
    return score


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]
