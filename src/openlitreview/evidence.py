from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .fulltext import FullTextResult
from .llm import LLMClient
from .prompts import EVIDENCE_SYSTEM, evidence_prompt
from .schemas import EvidenceCard, PaperRecord, TaskSpec, safe_resolve
from .storage import citation_key


class EvidenceExtractionError(RuntimeError):
    pass


async def extract_evidence_cards(
    task: TaskSpec,
    papers: list[PaperRecord],
    fulltexts: list[FullTextResult],
    client: LLMClient,
    output: Path,
    *,
    initial_cards: list[EvidenceCard] | None = None,
    initial_log: list[dict[str, Any]] | None = None,
) -> tuple[list[EvidenceCard], list[dict[str, Any]]]:
    fulltext_by_record = {
        result.record_id: result
        for result in fulltexts
        if result.status == "extracted" and result.text_path
    }
    allowed_record_ids = {paper.record_id for paper in papers[: task.search.target_fulltexts]}
    cards = [card for card in (initial_cards or []) if card.record_id in allowed_record_ids]
    log = [
        item
        for item in (initial_log or [])
        if not item.get("record_id") or item.get("record_id") in allowed_record_ids
    ]
    processed_record_ids = {card.record_id for card in cards}
    consecutive_failures = 0
    for paper in papers[: task.search.target_fulltexts]:
        if paper.record_id in processed_record_ids:
            continue
        if paper.publication_status in {"retracted", "withdrawn"}:
            log.append(
                {
                    "record_id": paper.record_id,
                    "status": "excluded_adverse_publication_status",
                    "publication_status": paper.publication_status,
                }
            )
            continue
        fulltext = fulltext_by_record.get(paper.record_id)
        verified = fulltext is not None
        if fulltext and fulltext.text_path:
            text = Path(fulltext.text_path).read_text(encoding="utf-8")
        else:
            text = paper.abstract or ""
        if len(text) < 200:
            log.append({"record_id": paper.record_id, "status": "insufficient_text"})
            continue
        text = text[:45_000]
        try:
            payload = await client.complete_json(
                model_alias=task.models.cheap_model,
                system=EVIDENCE_SYSTEM,
                prompt=evidence_prompt(paper, text, verified),
                max_output_tokens=6_000,
                temperature=0.0,
            )
            extracted = payload.get("evidence") or []
            paper_cards = _cards_from_payload(paper, payload, extracted, verified)
            cards.extend(paper_cards)
            consecutive_failures = 0
            log.append(
                {
                    "record_id": paper.record_id,
                    "citation_key": citation_key(paper),
                    "status": "ok",
                    "fulltext_verified": verified,
                    "cards": len(paper_cards),
                }
            )
        except Exception as exc:
            consecutive_failures += 1
            log.append(
                {
                    "record_id": paper.record_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            )
            if consecutive_failures >= 3:
                write_evidence_outputs(cards, log, papers, output)
                raise EvidenceExtractionError(
                    "Evidence extraction stopped after three consecutive model failures"
                ) from exc
    write_evidence_outputs(cards, log, papers, output)
    return cards, log


def load_evidence_seed(
    task_path: str | Path, relative_path: str | None
) -> tuple[
    list[EvidenceCard],
    list[dict[str, Any]],
    list[PaperRecord],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not relative_path:
        return [], [], [], [], {}
    seed_path = safe_resolve(Path(task_path).parent, relative_path)
    if not seed_path.is_file() or seed_path.is_symlink():
        raise FileNotFoundError(f"Evidence seed file not found: {seed_path.name}")
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evidence seed must contain a JSON object")
    raw_cards = payload.get("cards") or []
    raw_log = payload.get("log") or []
    raw_papers = payload.get("papers") or []
    raw_digest = payload.get("evidence_digest_batches") or []
    raw_writing_checkpoints = payload.get("writing_checkpoints") or {}
    if not all(isinstance(value, list) for value in (raw_cards, raw_log, raw_papers, raw_digest)):
        raise ValueError("Evidence seed cards, log, papers, and digest must be arrays")
    if not isinstance(raw_writing_checkpoints, dict):
        raise ValueError("Evidence seed writing_checkpoints must be an object")
    cards = [EvidenceCard.model_validate(card) for card in raw_cards]
    log = [dict(item) for item in raw_log if isinstance(item, dict)]
    papers = [PaperRecord.model_validate(paper) for paper in raw_papers]
    digest = [dict(item) for item in raw_digest if isinstance(item, dict)]
    return cards, log, papers, digest, dict(raw_writing_checkpoints)


def _cards_from_payload(
    paper: PaperRecord,
    payload: dict[str, Any],
    extracted: list[Any],
    verified: bool,
) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for index, item in enumerate(extracted[:12], start=1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        result = str(item.get("result") or "").strip()
        if not claim or not result:
            continue
        limitations = [
            str(value).strip() for value in item.get("limitations") or [] if str(value).strip()
        ]
        cards.append(
            EvidenceCard(
                evidence_id=f"{citation_key(paper)}_e{index}",
                record_id=paper.record_id,
                claim=claim,
                evidence_type="fulltext" if verified else "abstract",
                study_design=_optional_string(payload.get("study_design")),
                population=_optional_string(payload.get("population")),
                result=result,
                limitations=limitations,
                locator=_optional_string(item.get("locator"))
                or ("abstract" if not verified else None),
                fulltext_verified=verified,
                confidence=str(item.get("confidence") or "medium").lower(),
            )
        )
    return cards


def write_evidence_outputs(
    cards: list[EvidenceCard],
    log: list[dict[str, Any]],
    papers: list[PaperRecord],
    output: Path,
) -> None:
    evidence_dir = output / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paper_map = {paper.record_id: paper for paper in papers}
    (evidence_dir / "evidence_cards.json").write_text(
        json.dumps(
            [card.model_dump(mode="json") for card in cards],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "extraction_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "evidence_id",
        "citation_key",
        "title",
        "year",
        "evidence_type",
        "study_design",
        "population",
        "claim",
        "result",
        "limitations",
        "locator",
        "confidence",
    ]
    with (evidence_dir / "evidence_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for card in cards:
            paper = paper_map[card.record_id]
            writer.writerow(
                {
                    "evidence_id": card.evidence_id,
                    "citation_key": citation_key(paper),
                    "title": paper.title,
                    "year": paper.year or "",
                    "evidence_type": card.evidence_type,
                    "study_design": card.study_design or "",
                    "population": card.population or "",
                    "claim": card.claim,
                    "result": card.result,
                    "limitations": "; ".join(card.limitations),
                    "locator": card.locator or "",
                    "confidence": card.confidence,
                }
            )


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
