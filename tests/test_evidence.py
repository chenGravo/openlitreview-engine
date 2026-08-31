from __future__ import annotations

from typing import Any

import pytest

from openlitreview.evidence import (
    EvidenceExtractionError,
    extract_evidence_cards,
    load_evidence_seed,
)
from openlitreview.schemas import EvidenceCard, PaperRecord, TaskSpec


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
    assert client.model_aliases == ["deepseek-v4-flash"] * 3
    assert (tmp_path / "evidence" / "extraction_log.json").is_file()


@pytest.mark.asyncio
async def test_evidence_seed_skips_already_processed_paper(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={"enabled": True},
        search={"target_fulltexts": 10},
    )
    papers = [
        PaperRecord(record_id="seeded", title="Seeded paper", abstract="A" * 500),
        PaperRecord(record_id="new", title="New paper", abstract="B" * 500),
        PaperRecord(record_id="third", title="Third paper", abstract="C" * 500),
        PaperRecord(record_id="fourth", title="Fourth paper", abstract="D" * 500),
    ]
    seed = EvidenceCard(
        evidence_id="seed_e1",
        record_id="seeded",
        claim="Seed claim",
        evidence_type="abstract",
        result="Seed result",
    )
    client = FailingClient()

    with pytest.raises(EvidenceExtractionError, match="three consecutive"):
        await extract_evidence_cards(
            task,
            papers,
            [],
            client,
            tmp_path,
            initial_cards=[seed],
            initial_log=[{"record_id": "seeded", "status": "ok", "cards": 1}],
        )

    assert client.calls == 3


@pytest.mark.asyncio
async def test_orphan_seed_cards_are_filtered_before_outputs(tmp_path) -> None:
    task = TaskSpec(
        title="测试文献综述",
        research_question="测试研究问题是什么？",
        keywords=["test"],
        models={"enabled": True},
        search={"target_fulltexts": 10},
    )
    papers = [
        PaperRecord(record_id=f"p{index}", title=f"Paper {index}", abstract="A" * 500)
        for index in range(4)
    ]
    orphan = EvidenceCard(
        evidence_id="orphan_e1",
        record_id="not-in-current-papers",
        claim="Orphan claim",
        evidence_type="abstract",
        result="Orphan result",
    )

    with pytest.raises(EvidenceExtractionError, match="three consecutive"):
        await extract_evidence_cards(
            task,
            papers,
            [],
            FailingClient(),
            tmp_path,
            initial_cards=[orphan],
            initial_log=[{"record_id": orphan.record_id, "status": "ok"}],
        )

    assert "not-in-current-papers" not in (
        tmp_path / "evidence" / "evidence_cards.json"
    ).read_text(encoding="utf-8")


def test_evidence_seed_loads_paper_metadata(tmp_path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        """{
  "cards": [
    {"evidence_id":"e1","record_id":"p1","claim":"c",
     "evidence_type":"abstract","result":"r"}
  ],
  "log": [{"record_id":"p1","status":"ok"}],
  "papers": [{"record_id":"p1","title":"Paper one","doi":"10.1/test"}]
}
""",
        encoding="utf-8",
    )

    cards, log, papers = load_evidence_seed(tmp_path / "task.json", "seed.json")

    assert cards[0].record_id == "p1"
    assert log[0]["status"] == "ok"
    assert papers[0].doi == "10.1/test"
