from types import SimpleNamespace

from openlitreview.pipeline import _select_seed_papers
from openlitreview.schemas import PaperRecord


def _paper(record_id: str) -> PaperRecord:
    return PaperRecord(record_id=record_id, title=f"Paper {record_id}")


def test_seed_papers_without_cards_are_kept_for_new_extraction() -> None:
    pending = _paper("pending")
    processed = _paper("processed")
    cards = [SimpleNamespace(record_id="processed")]

    selected = _select_seed_papers([pending, processed], cards, target=2)

    assert [paper.record_id for paper in selected] == ["processed", "pending"]


def test_seed_paper_target_applies_after_processed_first_ordering() -> None:
    papers = [_paper("pending-a"), _paper("processed"), _paper("pending-b")]
    cards = [SimpleNamespace(record_id="processed")]

    selected = _select_seed_papers(papers, cards, target=2)

    assert [paper.record_id for paper in selected] == ["processed", "pending-a"]
