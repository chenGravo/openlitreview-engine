from __future__ import annotations

from typing import Any

import pytest

from openlitreview.schemas import EvidenceCard, PaperRecord, TaskSpec
from openlitreview.writer import generate_review


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0

    async def complete_json(self, **_: Any) -> dict[str, Any]:
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
        models={"enabled": True, "max_revision_rounds": 1},
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

    assert client.calls == 5
    assert "修订后表述" in markdown
    assert payload == revised
    assert reviewer == {"verdict": "pass", "issues": []}
    assert (tmp_path / "audit" / "independent_model_review_2.json").is_file()
