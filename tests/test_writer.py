from __future__ import annotations

from typing import Any

import pytest

from openlitreview.schemas import EvidenceCard, PaperRecord, TaskSpec
from openlitreview.writer import (
    _citation_aliases_from_digest,
    _normalize_part_citations,
    _paper_batches,
    _sanitize_batch_digest,
    generate_review,
)


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0
        self.model_aliases: list[str] = []

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.model_aliases.append(str(kwargs["model_alias"]))
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _draft(body: str) -> dict[str, Any]:
    return {
        "title": "测试综述",
        "abstract": "摘要",
        "keywords": ["测试"],
        "introduction": body,
        "sections": [{"heading": "结果", "body": body}],
        "conclusion": body,
        "limitations": "证据有限。[@ref_50c81ef030]",
    }


@pytest.mark.asyncio
async def test_failed_independent_review_triggers_revision_and_rereview(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={
            "enabled": True,
            "cheap_model": "deepseek-v4-pro",
            "primary_model": "deepseek-v4-pro",
            "perspective_model": "kimi-k2.6",
            "reviewer_model": "doubao-seed-2.1-pro",
            "allow_same_model_quality_checks": False,
            "max_revision_rounds": 1,
        },
    )
    paper = PaperRecord(record_id="p1", title="Test paper", doi="10.1/test")
    card = EvidenceCard(
        evidence_id="e1",
        record_id="p1",
        claim="测试主张",
        evidence_type="abstract",
        result="测试结果",
    )
    initial = _draft("初稿表述。[@ref_50c81ef030]")
    revised = _draft("修订后表述。[@ref_50c81ef030]")
    client = FakeClient(
        [
            {
                "source_summaries": [
                    {
                        "citation_key": "ref_50c81ef030",
                        "evidence_ids": ["e1"],
                        "supported_findings": ["测试结果"],
                    }
                ]
            },
            {
                "evidence_clusters": [],
                "contradictions": [],
                "outline_requirements": [],
            },
            {"central_argument": "测试", "sections": []},
            initial,
            {"verdict": "revise", "issues": [{"severity": "high"}]},
            revised,
            {"verdict": "pass", "issues": []},
        ]
    )
    (tmp_path / "audit").mkdir()

    markdown, payload, reviewer = await generate_review(task, [paper], [card], client, tmp_path)

    assert client.calls == 7
    assert client.model_aliases == [
        "kimi-k2.6",
        "kimi-k2.6",
        "deepseek-v4-pro",
        "deepseek-v4-pro",
        "doubao-seed-2.1-pro",
        "deepseek-v4-pro",
        "doubao-seed-2.1-pro",
    ]
    assert "修订后表述" in markdown
    assert payload == revised
    assert reviewer == {"verdict": "pass", "issues": []}
    assert (tmp_path / "audit" / "prewriting_perspective_audit.json").is_file()
    assert (tmp_path / "audit" / "independent_model_review_2.json").is_file()


@pytest.mark.asyncio
async def test_explicit_same_model_review_is_labeled_as_same_model(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={
            "enabled": True,
            "cheap_model": "deepseek-v4-flash",
            "primary_model": "kimi-k2.6",
            "perspective_model": "kimi-k2.6",
            "reviewer_model": "kimi-k2.6",
            "allow_same_model_quality_checks": True,
            "max_revision_rounds": 0,
        },
    )
    paper = PaperRecord(record_id="p1", title="Test paper", doi="10.1/test")
    card = EvidenceCard(
        evidence_id="e1",
        record_id="p1",
        claim="测试主张",
        evidence_type="abstract",
        result="测试结果",
    )
    client = FakeClient(
        [
            {
                "source_summaries": [
                    {
                        "citation_key": "ref_50c81ef030",
                        "evidence_ids": ["e1"],
                        "supported_findings": ["测试结果"],
                    }
                ]
            },
            {"evidence_clusters": [], "contradictions": []},
            {"central_argument": "测试", "sections": []},
            _draft("正文。[@ref_50c81ef030]"),
            {"verdict": "pass", "issues": []},
        ]
    )
    (tmp_path / "audit").mkdir()

    await generate_review(task, [paper], [card], client, tmp_path)

    assert client.model_aliases == ["kimi-k2.6"] * 5
    assert (tmp_path / "audit" / "evidence_digest_batches.json").is_file()
    assert (tmp_path / "audit" / "same_model_review.json").is_file()
    assert not (tmp_path / "audit" / "independent_model_review.json").exists()


@pytest.mark.asyncio
async def test_seeded_digest_batches_skip_completed_model_calls(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={
            "enabled": True,
            "primary_model": "kimi-k2.6",
            "perspective_model": "kimi-k2.6",
            "reviewer_model": "kimi-k2.6",
            "allow_same_model_quality_checks": True,
            "max_revision_rounds": 0,
        },
    )
    paper = PaperRecord(record_id="p1", title="Test paper", doi="10.1/test")
    card = EvidenceCard(
        evidence_id="e1",
        record_id="p1",
        claim="测试主张",
        evidence_type="abstract",
        result="测试结果",
    )
    digest = [
        {
            "batch_number": 1,
            "source_summaries": [
                {
                    "citation_key": "ref_50c81ef030",
                    "evidence_ids": ["e1"],
                    "supported_findings": ["测试结果"],
                }
            ],
        }
    ]
    client = FakeClient(
        [
            _draft("正文。[@ref_50c81ef030]"),
            {"verdict": "pass", "issues": []},
        ]
    )
    (tmp_path / "audit").mkdir()

    await generate_review(
        task,
        [paper],
        [card],
        client,
        tmp_path,
        initial_evidence_digest=digest,
        initial_writing_checkpoints={
            "perspective_audit": {
                "evidence_clusters": [],
                "contradictions": [],
            },
            "outline": {"central_argument": "测试", "sections": []},
        },
    )

    assert client.calls == 2
    assert client.model_aliases == ["kimi-k2.6"] * 2
    assert (tmp_path / "audit" / "writing_checkpoints.json").is_file()


@pytest.mark.asyncio
async def test_sectioned_draft_uses_bounded_calls_and_checkpoints(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={
            "enabled": True,
            "primary_model": "kimi-k2.6",
            "perspective_model": "kimi-k2.6",
            "reviewer_model": "kimi-k2.6",
            "allow_same_model_quality_checks": True,
            "max_revision_rounds": 0,
        },
    )
    paper = PaperRecord(record_id="p1", title="Test paper", doi="10.1/test")
    card = EvidenceCard(
        evidence_id="e1",
        record_id="p1",
        claim="测试主张",
        evidence_type="abstract",
        result="测试结果",
    )
    digest = [
        {
            "batch_number": 1,
            "source_summaries": [
                {
                    "citation_key": "ref_50c81ef030",
                    "evidence_ids": ["e1"],
                    "supported_findings": ["测试结果"],
                }
            ],
        }
    ]
    outline = {
        "central_argument": "测试",
        "sections": [
            {"heading": "引言", "purpose": "介绍", "evidence_ids": ["e1"]},
            {"heading": "核心结果", "purpose": "综合", "evidence_ids": ["e1"]},
            {"heading": "局限", "purpose": "局限", "evidence_ids": ["e1"]},
            {"heading": "结论", "purpose": "总结", "evidence_ids": ["e1"]},
        ],
    }
    structural = {
        "title": "测试综述",
        "abstract": "摘要",
        "keywords": ["测试"],
        "introduction": "引言。[@ref_50c81ef030]",
        "conclusion": "结论。[@ref_50c81ef030]",
        "limitations": "局限。[@ref_50c81ef030]",
    }
    client = FakeClient(
        [
            structural,
            {"heading": "核心结果", "body": "分节正文。[@ref_50c81ef030]"},
            {"verdict": "pass", "issues": []},
        ]
    )
    (tmp_path / "audit").mkdir()

    markdown, payload, _ = await generate_review(
        task,
        [paper],
        [card],
        client,
        tmp_path,
        initial_evidence_digest=digest,
        initial_writing_checkpoints={
            "perspective_audit": {"evidence_clusters": [], "contradictions": []},
            "outline": outline,
        },
    )

    assert client.calls == 3
    assert payload["sections"] == [{"heading": "核心结果", "body": "分节正文。[@ref_50c81ef030]"}]
    assert "分节正文" in markdown
    checkpoint = (tmp_path / "audit" / "writing_checkpoints.json").read_text(encoding="utf-8")
    assert '"initial"' in checkpoint
    assert '"sections"' in checkpoint


def test_digest_batches_preserve_every_source_and_drop_unknown_ids() -> None:
    packet = [
        {
            "citation_key": f"ref_{index}",
            "paper_title": f"Paper {index}",
            "paper_year": 2020 + index,
            "evidence_id": f"e{index}",
            "claim": "claim",
            "result": "result",
            "limitations": ["limited"],
        }
        for index in range(10)
    ]
    batches = _paper_batches(packet, max_papers=8)

    assert [len(batch) for batch in batches] == [8, 2]
    digest = _sanitize_batch_digest(
        {
            "source_summaries": [
                {
                    "citation_key": "ref_0",
                    "evidence_ids": ["e0", "invented"],
                    "supported_findings": ["supported"],
                },
                {"citation_key": "invented", "evidence_ids": ["invented"]},
            ],
            "cross_source_observations": [
                {"observation": "valid", "evidence_ids": ["e0", "invented"]},
                {"observation": "unsupported", "evidence_ids": ["invented"]},
            ],
        },
        batches[0],
        batch_number=1,
    )

    assert len(digest["source_summaries"]) == 8
    assert digest["source_summaries"][0]["evidence_ids"] == ["e0"]
    assert digest["cross_source_observations"] == [{"observation": "valid", "evidence_ids": ["e0"]}]


def test_evidence_card_citations_are_mapped_to_verified_paper_keys() -> None:
    digest = [
        {
            "source_summaries": [
                {
                    "citation_key": "ref_50c81ef030",
                    "evidence_ids": ["ref_50c81ef030_e1", "ref_50c81ef030_e10"],
                }
            ]
        }
    ]
    payload = {
        "introduction": "引言。[@ref_50c81ef030_e1]",
        "sections": [
            {"body": "正文。[@ref_50c81ef030_e99; @ref_50c81ef030_e1]"}
        ],
    }

    normalized = _normalize_part_citations(payload, _citation_aliases_from_digest(digest))

    assert normalized["introduction"] == "引言。[@ref_50c81ef030]"
    assert normalized["sections"][0]["body"] == "正文。[@ref_50c81ef030]"
