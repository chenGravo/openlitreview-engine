from __future__ import annotations

from typing import Any

import pytest

from openlitreview.evidence import EvidenceExtractionError, extract_evidence_cards
from openlitreview.schemas import PaperRecord, TaskSpec


class FailingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.model_aliases: list[str] = []

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.model_aliases.append(str(kwargs["model_alias"]))
        raise RuntimeError("synthetic provider failure")


@pytest.mark.asyncio
async def test_evidence_extraction_stops_after_three_consecutive_failures(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={"enabled": True},
        search={"target_fulltexts": 10},
    )
    papers = [
        PaperRecord(
            record_id=f"p{index}",
            title=f"Test paper {index}",
            abstract="A" * 500,
        )
        for index in range(10)
    ]
    client = FailingClient()

    with pytest.raises(EvidenceExtractionError, match="three consecutive"):
        await extract_evidence_cards(task, papers, [], client, tmp_path)

    assert client.calls == 3
    assert client.model_aliases == ["deepseek-v4-pro"] * 3
    assert (tmp_path / "evidence" / "extraction_log.json").is_file()
