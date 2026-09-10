from types import SimpleNamespace

from openlitreview.pipeline import _select_evidence_papers
from openlitreview.schemas import PaperRecord


def _paper(
    record_id: str,
    *,
    year: int,
    citations: int,
    primary_match: float,
) -> PaperRecord:
    return PaperRecord(
        record_id=record_id,
        title=f"Paper {record_id}",
        year=year,
        citation_count=citations,
        rank_breakdown={
            "primary_scope_match": primary_match,
            "scope_match": primary_match,
            "relevance": primary_match,
        },
    )


def test_evidence_selection_reserves_foundational_and_rejects_noise_when_possible() -> None:
    foundational = _paper(
        "foundational", year=1970, citations=10_000, primary_match=1.0
    )
    recent = [
        _paper(f"relevant-{index}", year=2020 + index % 5, citations=10, primary_match=1.0)
        for index in range(12)
    ]
    noise = _paper("noise", year=2025, citations=1_000_000, primary_match=0.0)
    papers = [noise, *recent, foundational]
    fulltexts = [
        SimpleNamespace(record_id=paper.record_id, status="extracted")
        for paper in [noise, *recent]
    ]

    selected = _select_evidence_papers(papers, fulltexts, target=10)
    selected_ids = {paper.record_id for paper in selected}

    assert "foundational" in selected_ids
    assert "noise" not in selected_ids
    selected_foundation = next(
        paper for paper in selected if paper.record_id == "foundational"
    )
    assert "foundational_priority" in selected_foundation.quality_flags

