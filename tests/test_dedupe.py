from openlitreview.dedupe import deduplicate_papers
from openlitreview.schemas import PaperRecord


def test_merge_same_doi_keeps_richer_metadata() -> None:
    papers = [
        PaperRecord(
            record_id="a",
            title="Example Intervention in Community Settings",
            doi="https://doi.org/10.1000/ABC",
            source_names=["semantic_scholar"],
            citation_count=4,
        ),
        PaperRecord(
            record_id="b",
            title="Example Intervention in Community Settings",
            abstract="A sufficiently informative abstract.",
            doi="10.1000/abc",
            source_names=["crossref"],
            citation_count=10,
        ),
    ]
    merged = deduplicate_papers(papers)
    assert len(merged) == 1
    assert merged[0].doi == "10.1000/abc"
    assert merged[0].citation_count == 10
    assert merged[0].abstract
    assert set(merged[0].source_names) == {"semantic_scholar", "crossref"}


def test_near_identical_title_and_nearby_year_merge() -> None:
    papers = [
        PaperRecord(record_id="a", title="Effects of example-intervention", year=2020),
        PaperRecord(record_id="b", title="Effects of example intervention", year=2021),
    ]
    assert len(deduplicate_papers(papers)) == 1
