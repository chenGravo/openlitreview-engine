from __future__ import annotations

from typing import Any

import pytest

from openlitreview.schemas import EvidenceCard, PaperRecord, TaskSpec
from openlitreview.writer import generate_review


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

    markdown, payload, reviewer = await generate_review(
        task, [paper], [card], client, tmp_path
    )

    assert client.calls == 6
    assert client.model_aliases == [
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
            {"evidence_clusters": [], "contradictions": []},
            {"central_argument": "测试", "sections": []},
            _draft("正文。[@ref_50c81ef030]"),
            {"verdict": "pass", "issues": []},
        ]
    )
    (tmp_path / "audit").mkdir()

    await generate_review(task, [paper], [card], client, tmp_path)

    assert client.model_aliases == ["kimi-k2.6"] * 4
    assert (tmp_path / "audit" / "same_model_review.json").is_file()
    assert not (tmp_path / "audit" / "independent_model_review.json").exists()
