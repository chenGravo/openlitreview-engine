from openlitreview.audit import _unfinished_section_endings, audit_run
from openlitreview.schemas import EvidenceCard, PaperRecord, SearchRun, TaskSpec


def test_unknown_citation_blocks_draft(tmp_path) -> None:
    task = TaskSpec(
        title="Example intervention",
        research_question="How does the example intervention affect outcomes?",
        keywords=["example intervention"],
        search={"minimum_independent_sources": 2},
    )
    run = SearchRun(
        task_id="test",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        queries=["example intervention"],
        source_status={
            "semantic_scholar": {"status": "ok", "records": 1},
            "crossref": {"status": "ok", "records": 1},
        },
        papers=[PaperRecord(record_id="paper", title="A paper", doi="10.1/test")],
    )
    card = EvidenceCard(
        evidence_id="e1",
        record_id="paper",
        claim="A claim",
        evidence_type="abstract",
        result="A result",
    )
    report = audit_run(
        task,
        run,
        [card],
        "## 结果\n\n这是一段足够长的测试文本。[@invented_reference]",
        {"verdict": "pass"},
        tmp_path,
    )
    assert report["status"] == "blocked"
    assert report["unknown_citations"] == ["invented_reference"]


def test_unfinished_section_endings_detect_truncated_prose() -> None:
    markdown = """## 摘要

摘要正文。

**关键词：** 测试；审计

## 1 完整章节

完整论述。[@known]

## 2 截断章节

上述发现不可直接解读为

## 参考文献

[1] Example.
"""
    assert _unfinished_section_endings(markdown) == ["2 截断章节"]


def test_incomplete_publication_status_checks_block_quality_gate(tmp_path) -> None:
    task = TaskSpec(
        title="Publication status audit",
        research_question="What does the evidence show?",
        keywords=["evidence"],
        search={"minimum_independent_sources": 2},
        quality={
            "minimum_retained_records": 5,
            "minimum_evidence_papers": 3,
            "minimum_fulltext_verified_papers": 0,
            "minimum_cited_papers": 3,
        },
    )
    papers = [
        PaperRecord(
            record_id=f"p{index}",
            title=f"Paper {index}",
            publication_status="no_adverse_update_found" if index == 0 else "unchecked",
        )
        for index in range(5)
    ]
    run = SearchRun(
        task_id="integrity-test",
        started_at="2026-01-01T00:00:00Z",
        queries=["evidence"],
        source_status={
            "crossref": {"status": "ok", "records": 3},
            "semantic_scholar": {"status": "ok", "records": 2},
        },
        papers=papers,
    )
    cards = [
        EvidenceCard(
            evidence_id=f"e{index}",
            record_id=f"p{index}",
            claim="Claim",
            evidence_type="abstract",
            result="Result",
        )
        for index in range(3)
    ]

    report = audit_run(task, run, cards, None, None, tmp_path)

    assert report["status"] == "blocked"
    assert report["publication_status_checked_papers"] == 1
    assert report["publication_status_counts"] == {
        "no_adverse_update_found": 1,
        "unchecked": 2,
    }
    assert any(
        item["code"] == "insufficient_publication_status_checks"
        for item in report["findings"]
    )


def test_unlicensed_document_cannot_satisfy_fulltext_quality_gate(tmp_path) -> None:
    task = TaskSpec(
        title="Full text license audit",
        research_question="What does the evidence show?",
        keywords=["evidence"],
        search={"minimum_independent_sources": 2},
        quality={
            "minimum_retained_records": 5,
            "minimum_evidence_papers": 3,
            "minimum_fulltext_verified_papers": 1,
            "minimum_cited_papers": 3,
        },
    )
    papers = [
        PaperRecord(
            record_id=f"p{index}",
            title=f"Paper {index}",
            publication_status="no_adverse_update_found",
            open_access_pdf_url="https://example.org/article.pdf" if index == 0 else None,
            open_access_license=None,
        )
        for index in range(5)
    ]
    run = SearchRun(
        task_id="license-test",
        started_at="2026-01-01T00:00:00Z",
        queries=["evidence"],
        source_status={
            "crossref": {"status": "ok", "records": 3},
            "semantic_scholar": {"status": "ok", "records": 2},
        },
        papers=papers,
    )
    cards = [
        EvidenceCard(
            evidence_id=f"e{index}",
            record_id=f"p{index}",
            claim="Claim",
            evidence_type="fulltext" if index == 0 else "abstract",
            result="Result",
            fulltext_verified=index == 0,
        )
        for index in range(3)
    ]

    report = audit_run(task, run, cards, None, None, tmp_path)

    assert report["status"] == "blocked"
    assert report["fulltext_license_verified_papers"] == 0
    assert any(item["code"] == "fulltext_license_not_verified" for item in report["findings"])


def test_bibliographic_metadata_does_not_satisfy_substantive_evidence_minimum(
    tmp_path,
) -> None:
    task = TaskSpec(
        title="Metadata evidence audit",
        research_question="What does the evidence show?",
        keywords=["evidence"],
        search={"minimum_independent_sources": 2},
        quality={
            "minimum_retained_records": 5,
            "minimum_evidence_papers": 3,
            "minimum_fulltext_verified_papers": 0,
            "minimum_cited_papers": 3,
        },
    )
    papers = [
        PaperRecord(
            record_id=f"p{index}",
            title=f"Paper {index}",
            publication_status="no_adverse_update_found",
        )
        for index in range(5)
    ]
    run = SearchRun(
        task_id="metadata-test",
        started_at="2026-01-01T00:00:00Z",
        queries=["evidence"],
        source_status={
            "crossref": {"status": "ok", "records": 3},
            "semantic_scholar": {"status": "ok", "records": 2},
        },
        papers=papers,
    )
    cards = [
        EvidenceCard(
            evidence_id=f"e{index}",
            record_id=f"p{index}",
            claim="Claim",
            evidence_type=("bibliographic_metadata" if index == 2 else "abstract"),
            result="Result",
        )
        for index in range(3)
    ]

    report = audit_run(task, run, cards, None, None, tmp_path)

    assert report["evidence_papers"] == 2
    assert report["bibliographic_metadata_papers"] == 1
    assert any(item["code"] == "insufficient_evidence_papers" for item in report["findings"])
