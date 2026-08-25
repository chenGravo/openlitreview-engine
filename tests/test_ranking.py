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
