from types import SimpleNamespace

from openlitreview.pipeline import (
    _merge_papers,
    _select_new_evidence_papers,
    _select_pending_seed_papers,
    _select_seed_papers,
)
from openlitreview.schemas import PaperRecord


def _paper(record_id: str) -> PaperRecord:
    return PaperRecord(record_id=record_id, title=f"Paper {record_id}")


def test_seed_papers_with_cards_are_reused_without_new_extraction() -> None:
    pending = _paper("pending")
    processed = _paper("processed")
    cards = [SimpleNamespace(record_id="processed")]

    selected = _select_seed_papers([pending, processed], cards, target=2)

    assert [paper.record_id for paper in selected] == ["processed"]


def test_metadata_only_seed_papers_are_queued_for_fulltext_collection() -> None:
    papers = [_paper("pending-a"), _paper("processed"), _paper("pending-b")]
    cards = [SimpleNamespace(record_id="processed")]

    pending = _select_pending_seed_papers(papers, cards, target=2)

    assert [paper.record_id for paper in pending] == ["pending-a", "pending-b"]


def test_processed_and_pending_seeds_precede_search_results() -> None:
    processed = _paper("processed")
    pending = _paper("pending")
    search = _paper("search")

    merged = _merge_papers([processed, pending], [search])

    assert [paper.record_id for paper in merged] == ["processed", "pending", "search"]


def test_pending_seeds_are_selected_before_unseeded_fulltexts() -> None:
    pending_a = _paper("pending-a")
    pending_b = _paper("pending-b")
    search = _paper("search")
    fulltexts = [SimpleNamespace(record_id="search", status="extracted")]

    selected = _select_new_evidence_papers(
        [pending_a, pending_b], [pending_a, pending_b, search], fulltexts, target=2
    )

    assert [paper.record_id for paper in selected] == ["pending-a", "pending-b"]
