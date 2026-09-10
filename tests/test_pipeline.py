import pytest

import openlitreview.pipeline as pipeline_module
from openlitreview.schemas import PaperRecord


@pytest.mark.asyncio
async def test_evidence_papers_with_unresolved_doi_are_rechecked(monkeypatch) -> None:
    checked_ids: list[str] = []

    async def fake_check(papers: list[PaperRecord]) -> list[PaperRecord]:
        checked_ids.extend(paper.record_id for paper in papers)
        return [
            paper.model_copy(update={"publication_status": "no_adverse_update_found"})
            for paper in papers
        ]

    monkeypatch.setattr(pipeline_module, "check_publication_updates", fake_check)
    papers = [
        PaperRecord(
            record_id="already-checked",
            title="Already checked",
            doi="10.1/checked",
            publication_status="no_adverse_update_found",
        ),
        PaperRecord(
            record_id="failed",
            title="Failed previously",
            doi="10.1/failed",
            publication_status="check_failed",
        ),
        PaperRecord(
            record_id="unchecked",
            title="Unchecked",
            doi="10.1/unchecked",
        ),
        PaperRecord(record_id="no-doi", title="No DOI"),
    ]

    refreshed = await pipeline_module._ensure_publication_status_checked(papers)

    assert checked_ids == ["failed", "unchecked"]
    assert [paper.record_id for paper in refreshed] == [paper.record_id for paper in papers]
    assert [paper.publication_status for paper in refreshed] == [
        "no_adverse_update_found",
        "no_adverse_update_found",
        "no_adverse_update_found",
        "unchecked",
    ]
