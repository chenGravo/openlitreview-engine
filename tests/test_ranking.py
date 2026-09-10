from openlitreview.ranking import rank_papers
from openlitreview.schemas import PaperRecord, TaskSpec


def test_relevant_paper_ranks_above_unrelated_highly_cited_paper() -> None:
    task = TaskSpec(
        title="Example intervention",
        research_question="How does the example intervention affect the target outcome?",
        keywords=["example intervention", "target outcome"],
    )
    papers = [
        PaperRecord(
            record_id="relevant",
            title="Example intervention effects on the target outcome",
            abstract="The example intervention improved the prespecified target outcome.",
            year=2024,
            citation_count=5,
        ),
        PaperRecord(
            record_id="unrelated",
            title="A highly cited method for galaxy spectroscopy",
            abstract="Astronomy and telescope calibration.",
            year=2010,
            citation_count=10_000,
        ),
    ]
    ranked = rank_papers(papers, task)
    assert ranked[0].record_id == "relevant"


def test_primary_scope_phrase_demotes_generic_token_overlap() -> None:
    task = TaskSpec(
        title="Grounded theory methods",
        research_question="How has grounded theory methodology developed?",
        keywords=["grounded theory", "theoretical sampling"],
    )
    papers = [
        PaperRecord(
            record_id="method",
            title="Grounded theory methodology and theoretical sampling",
            abstract="A methodological discussion of grounded theory.",
            year=2000,
            citation_count=25,
        ),
        PaperRecord(
            record_id="physics",
            title="A theoretical sampling method for particle transport",
            abstract="A high-dimensional physics sampling algorithm.",
            year=2025,
            citation_count=50_000,
        ),
    ]

    ranked = rank_papers(papers, task)

    assert ranked[0].record_id == "method"
    assert "primary_scope_missing" in ranked[1].quality_flags


def test_rank_papers_strictly_enforces_configured_year_range() -> None:
    task = TaskSpec(
        title="Date-bounded review",
        research_question="What does the target literature report?",
        keywords=["target literature"],
        year_from=2000,
        year_to=2020,
    )
    papers = [
        PaperRecord(record_id="old", title="Target literature", year=1999),
        PaperRecord(record_id="inside", title="Target literature", year=2020),
        PaperRecord(record_id="future", title="Target literature", year=2021),
        PaperRecord(record_id="unknown", title="Target literature"),
    ]

    ranked = rank_papers(papers, task)

    assert [paper.record_id for paper in ranked] == ["inside"]
