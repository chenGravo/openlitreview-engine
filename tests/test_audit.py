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
