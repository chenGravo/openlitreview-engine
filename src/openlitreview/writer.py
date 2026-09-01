from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .prompts import WRITING_SYSTEM
from .schemas import EvidenceCard, PaperRecord, TaskSpec
from .storage import citation_key

OUTLINE_SYSTEM = """You design a Chinese narrative literature-review outline from an audited
evidence set. Cover foundational concepts, major evidence clusters, disagreements, limitations,
and future directions. Do not invent citations or facts. Return strict JSON only."""

REVIEW_SYSTEM = """You are an independent academic evidence auditor. Compare the supplied
Chinese draft against the evidence cards. Identify unsupported, overstated, causal, numerical,
medical-safety, contradictory, or missing-counterevidence claims. Do not rewrite for style and do
not invent sources. Return strict JSON only."""

PERSPECTIVE_SYSTEM = """You are the pre-writing cross-evidence perspective auditor for a Chinese
academic literature review. Use only the supplied evidence cards. Identify competing explanations,
contradictory or null findings, population and subgroup boundaries, education-versus-medical
boundaries, safety uncertainty, missing perspectives, and limitations that the outline and draft
must preserve. Do not invent sources, facts, identifiers, or certainty. Return strict JSON only."""

REVISION_SYSTEM = """You revise a Chinese narrative academic literature review after an
independent evidence audit. Correct every reported issue using only the supplied evidence cards.
Preserve supported nuance, disagreements, null results, limitations, and Pandoc citation keys.
Never invent facts or citations. Return the complete revised review as strict JSON only."""

DIGEST_SYSTEM = """You create a loss-minimizing evidence digest for one batch of an audited
academic evidence set. Produce exactly one source summary for every supplied citation key. Retain
quantitative results, null findings, disagreements, population and intervention boundaries,
study-design strength, medical-safety uncertainty, and limitations. Use only supplied evidence
IDs and citation keys. Do not infer missing results or invent facts, sources, identifiers, or
certainty. Return strict JSON only."""


async def generate_review(
    task: TaskSpec,
    papers: list[PaperRecord],
    cards: list[EvidenceCard],
    client: LLMClient,
    output: Path,
    *,
    initial_evidence_digest: list[dict[str, Any]] | None = None,
    initial_writing_checkpoints: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    raw_evidence_packet = _evidence_packet(papers, cards)
    evidence_digest = await _build_evidence_digest(
        task,
        raw_evidence_packet,
        client,
        output,
        initial_digest=initial_evidence_digest,
    )
    writing_checkpoints = initial_writing_checkpoints or {}
    seeded_perspective = writing_checkpoints.get("perspective_audit")
    perspective_payload = (
        dict(seeded_perspective)
        if isinstance(seeded_perspective, dict)
        else await _audit_perspectives(task, evidence_digest, client)
    )
    perspective_path = output / "audit" / "prewriting_perspective_audit.json"
    perspective_path.write_text(
        json.dumps(perspective_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    seeded_outline = writing_checkpoints.get("outline")
    outline_payload = (
        dict(seeded_outline)
        if isinstance(seeded_outline, dict)
        else await client.complete_json(
            model_alias=task.models.primary_model,
            system=OUTLINE_SYSTEM,
            prompt=(
                "Return schema: "
                '{"central_argument":"","sections":[{"heading":"","purpose":"",'
                '"evidence_ids":[]}],"required_disagreements":[],'
                '"limitations_to_state":[]}\n'
                + json.dumps(
                    {
                        "task": _task_payload(task),
                        "evidence_digest": evidence_digest,
                        "perspective_audit": perspective_payload,
                    },
                    ensure_ascii=False,
                )
            ),
            max_output_tokens=4_000,
            temperature=0.1,
        )
    )
    draft_dir = output / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    (draft_dir / "outline.json").write_text(
        json.dumps(outline_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "audit" / "writing_checkpoints.json").write_text(
        json.dumps(
            {
                "perspective_audit": perspective_payload,
                "outline": outline_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    draft_payload = await client.complete_json(
        model_alias=task.models.primary_model,
        system=WRITING_SYSTEM,
        prompt=(
            "Write the complete review. Target Chinese characters: "
            f"{task.output.target_chinese_characters}. Return schema: "
            '{"title":"","abstract":"","keywords":[],"introduction":"",'
            '"sections":[{"heading":"","body":""}],"conclusion":"",'
            '"limitations":""}. Do not include a manually written reference list; Pandoc will '
            "render it from verified metadata.\n"
            + json.dumps(
                {
                    "task": _task_payload(task),
                    "approved_outline": outline_payload,
                    "evidence_digest": evidence_digest,
                    "perspective_audit": perspective_payload,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=min(20_000, max(6_000, task.output.target_chinese_characters * 2)),
        temperature=task.models.temperature,
    )
    markdown = render_review_markdown(draft_payload)
    (draft_dir / "review.json").write_text(
        json.dumps(draft_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (draft_dir / "review.md").write_text(markdown, encoding="utf-8")

    reviewer_payload = None
    if task.models.allow_second_model_review:
        review_prefix = (
            "same_model_review"
            if task.models.primary_model == task.models.reviewer_model
            else "independent_model_review"
        )
        for review_round in range(task.models.max_revision_rounds + 1):
            reviewer_payload = await _review_draft(task, evidence_digest, markdown, client)
            review_path = output / "audit" / f"{review_prefix}_{review_round + 1}.json"
            review_path.write_text(
                json.dumps(reviewer_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if str(reviewer_payload.get("verdict") or "").lower() == "pass":
                break
            if review_round >= task.models.max_revision_rounds:
                break
            draft_payload = await client.complete_json(
                model_alias=task.models.primary_model,
                system=REVISION_SYSTEM,
                prompt=(
                    "Return the same complete review schema: "
                    '{"title":"","abstract":"","keywords":[],"introduction":"",'
                    '"sections":[{"heading":"","body":""}],"conclusion":"",'
                    '"limitations":""}.\n'
                    + json.dumps(
                        {
                            "task": _task_payload(task),
                            "evidence_digest": evidence_digest,
                            "current_draft": markdown,
                            "independent_audit": reviewer_payload,
                        },
                        ensure_ascii=False,
                    )
                ),
                max_output_tokens=min(
                    20_000, max(6_000, task.output.target_chinese_characters * 2)
                ),
                temperature=0.1,
            )
            markdown = render_review_markdown(draft_payload)
            (draft_dir / f"review_revision_{review_round + 1}.json").write_text(
                json.dumps(draft_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (draft_dir / "review.json").write_text(
                json.dumps(draft_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (draft_dir / "review.md").write_text(markdown, encoding="utf-8")
        if reviewer_payload is not None:
            (output / "audit" / f"{review_prefix}.json").write_text(
                json.dumps(reviewer_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return markdown, draft_payload, reviewer_payload


async def _audit_perspectives(
    task: TaskSpec,
    evidence_digest: list[dict[str, Any]],
    client: LLMClient,
) -> dict[str, Any]:
    model_alias = task.models.perspective_model
    return await client.complete_json(
        model_alias=model_alias,
        system=PERSPECTIVE_SYSTEM,
        prompt=(
            "Return schema: "
            '{"evidence_clusters":[{"label":"","evidence_ids":[]}],'
            '"contradictions":[{"description":"","evidence_ids":[]}],'
            '"missing_or_underrepresented_perspectives":[],"population_boundaries":[],'
            '"education_medical_boundaries":[],"safety_uncertainties":[],'
            '"required_limitations":[],"outline_requirements":[]}\n'
            + json.dumps(
                {
                    "task": _task_payload(task),
                    "evidence_digest": evidence_digest,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=6_000,
        temperature=0.2,
    )


async def _review_draft(
    task: TaskSpec,
    evidence_digest: list[dict[str, Any]],
    markdown: str,
    client: LLMClient,
) -> dict[str, Any]:
    return await client.complete_json(
        model_alias=task.models.reviewer_model,
        system=REVIEW_SYSTEM,
        prompt=(
            "Return schema: "
            '{"verdict":"pass|revise|reject","issues":[{"severity":"high|medium|low",'
            '"location":"","problem":"","evidence_ids":[],"required_action":""}],'
            '"missing_perspectives":[],"citation_problems":[]}\n'
            + json.dumps(
                {
                    "task": _task_payload(task),
                    "evidence_digest": evidence_digest,
                    "draft_markdown": markdown,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=6_000,
        temperature=0.0,
    )


async def _build_evidence_digest(
    task: TaskSpec,
    evidence_packet: list[dict[str, Any]],
    client: LLMClient,
    output: Path,
    *,
    initial_digest: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compress all evidence hierarchically without exposing one oversized prompt."""
    batches = _paper_batches(evidence_packet, max_papers=8)
    digests: list[dict[str, Any]] = []
    checkpoint_path = output / "audit" / "evidence_digest_batches.json"
    for batch_number, (batch, seed) in enumerate(
        zip(batches, initial_digest or [], strict=False), start=1
    ):
        if int(seed.get("batch_number") or 0) != batch_number:
            break
        digests.append(_sanitize_batch_digest(seed, batch, batch_number=batch_number))
    if digests:
        checkpoint_path.write_text(
            json.dumps(digests, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for batch_number in range(len(digests) + 1, len(batches) + 1):
        batch = batches[batch_number - 1]
        payload = await client.complete_json(
            model_alias=task.models.perspective_model,
            system=DIGEST_SYSTEM,
            prompt=(
                "Return schema: "
                '{"source_summaries":[{"citation_key":"","evidence_ids":[],'
                '"design_and_population":"","supported_findings":[],'
                '"quantitative_results":[],"null_or_conflicting_findings":[],'
                '"limitations":[]}],"cross_source_observations":'
                '[{"observation":"","evidence_ids":[]}]}\n'
                "For each source summary, use at most 5 evidence_ids and at most 3 items "
                "in each list. Keep every list item under 80 words. Be concise but retain "
                "reported effect sizes, uncertainty, null results, safety boundaries, and "
                "limitations.\n"
                + json.dumps(
                    {
                        "task": _task_payload(task),
                        "batch_number": batch_number,
                        "batch_count": len(batches),
                        "sources": batch,
                    },
                    ensure_ascii=False,
                )
            ),
            max_output_tokens=16_000,
            temperature=0.0,
        )
        digests.append(_sanitize_batch_digest(payload, batch, batch_number=batch_number))
        checkpoint_path.write_text(
            json.dumps(digests, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return digests


def _paper_batches(
    evidence_packet: list[dict[str, Any]], *, max_papers: int
) -> list[list[dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in evidence_packet:
        key = str(item.get("citation_key") or "").strip()
        if not key:
            continue
        source = by_key.setdefault(
            key,
            {
                "citation_key": key,
                "paper_title": item.get("paper_title"),
                "paper_year": item.get("paper_year"),
                "evidence": [],
            },
        )
        source["evidence"].append(
            {
                "evidence_id": item.get("evidence_id"),
                "study_design": item.get("study_design"),
                "population": item.get("population"),
                "claim": item.get("claim"),
                "result": item.get("result"),
                "limitations": item.get("limitations"),
                "evidence_type": item.get("evidence_type"),
                "locator": item.get("locator"),
                "confidence": item.get("confidence"),
            }
        )
    sources = list(by_key.values())
    return [sources[index : index + max_papers] for index in range(0, len(sources), max_papers)]


def _sanitize_batch_digest(
    payload: dict[str, Any],
    batch: list[dict[str, Any]],
    *,
    batch_number: int,
) -> dict[str, Any]:
    allowed = {str(source["citation_key"]): source for source in batch}
    allowed_evidence = {
        str(item.get("evidence_id"))
        for source in batch
        for item in source.get("evidence") or []
        if item.get("evidence_id")
    }
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload.get("source_summaries") or []:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("citation_key") or "").strip()
        if key not in allowed or key in seen:
            continue
        source = allowed[key]
        summaries.append(
            {
                "citation_key": key,
                "paper_title": source.get("paper_title"),
                "paper_year": source.get("paper_year"),
                "evidence_ids": _allowed_values(raw.get("evidence_ids"), allowed_evidence),
                "design_and_population": str(raw.get("design_and_population") or "").strip(),
                "supported_findings": _string_list(raw.get("supported_findings")),
                "quantitative_results": _string_list(raw.get("quantitative_results")),
                "null_or_conflicting_findings": _string_list(
                    raw.get("null_or_conflicting_findings")
                ),
                "limitations": _string_list(raw.get("limitations")),
            }
        )
        seen.add(key)
    for key, source in allowed.items():
        if key not in seen:
            summaries.append(_fallback_source_summary(source))

    observations: list[dict[str, Any]] = []
    for raw in payload.get("cross_source_observations") or []:
        if not isinstance(raw, dict):
            continue
        observation = str(raw.get("observation") or "").strip()
        evidence_ids = _allowed_values(raw.get("evidence_ids"), allowed_evidence)
        if observation and evidence_ids:
            observations.append({"observation": observation, "evidence_ids": evidence_ids})
    return {
        "batch_number": batch_number,
        "source_summaries": summaries,
        "cross_source_observations": observations,
    }


def _fallback_source_summary(source: dict[str, Any]) -> dict[str, Any]:
    evidence = [item for item in source.get("evidence") or [] if isinstance(item, dict)]
    selected = evidence[:4]
    findings = []
    for item in selected:
        claim = str(item.get("claim") or "").strip()
        result = str(item.get("result") or "").strip()
        finding = "；".join(part for part in (claim, result) if part)
        if finding:
            findings.append(finding)
    limitations = list(
        dict.fromkeys(text for item in selected for text in _string_list(item.get("limitations")))
    )[:8]
    designs = list(
        dict.fromkeys(
            str(item.get("study_design") or "").strip()
            for item in evidence
            if str(item.get("study_design") or "").strip()
        )
    )
    populations = list(
        dict.fromkeys(
            str(item.get("population") or "").strip()
            for item in evidence
            if str(item.get("population") or "").strip()
        )
    )
    return {
        "citation_key": source.get("citation_key"),
        "paper_title": source.get("paper_title"),
        "paper_year": source.get("paper_year"),
        "evidence_ids": [
            str(item.get("evidence_id")) for item in selected if item.get("evidence_id")
        ],
        "design_and_population": "；".join([*designs[:3], *populations[:3]]),
        "supported_findings": findings,
        "quantitative_results": [],
        "null_or_conflicting_findings": [],
        "limitations": limitations,
    }


def _allowed_values(value: Any, allowed: set[str]) -> list[str]:
    return [item for item in _string_list(value) if item in allowed]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def render_review_markdown(payload: dict[str, Any]) -> str:
    title = str(payload.get("title") or "文献综述").strip()
    abstract = str(payload.get("abstract") or "").strip()
    keywords = [str(value).strip() for value in payload.get("keywords") or [] if str(value).strip()]
    introduction = str(payload.get("introduction") or "").strip()
    sections = payload.get("sections") or []
    conclusion = str(payload.get("conclusion") or "").strip()
    limitations = str(payload.get("limitations") or "").strip()
    parts = [
        "---",
        f'title: "{title.replace(chr(34), chr(39))}"',
        'lang: "zh-CN"',
        'reference-section-title: "参考文献"',
        "link-citations: true",
        "---",
        "",
        "## 摘要",
        "",
        abstract,
        "",
        f"**关键词：** {'；'.join(keywords)}",
        "",
        "## 引言",
        "",
        introduction,
    ]
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        body = str(section.get("body") or "").strip()
        if heading and body:
            parts.extend(["", f"## {heading}", "", body])
    parts.extend(["", "## 结论", "", conclusion, "", "## 局限", "", limitations, ""])
    return "\n".join(parts)


def _evidence_packet(papers: list[PaperRecord], cards: list[EvidenceCard]) -> list[dict[str, Any]]:
    paper_map = {paper.record_id: paper for paper in papers}
    packet: list[dict[str, Any]] = []
    for card in cards:
        paper = paper_map.get(card.record_id)
        if not paper:
            continue
        packet.append(
            {
                "evidence_id": card.evidence_id,
                "citation_key": citation_key(paper),
                "paper_title": paper.title,
                "paper_year": paper.year,
                "study_design": card.study_design,
                "population": card.population,
                "claim": card.claim,
                "result": card.result,
                "limitations": card.limitations,
                "evidence_type": card.evidence_type,
                "locator": card.locator,
                "confidence": card.confidence,
            }
        )
    return packet


def _task_payload(task: TaskSpec) -> dict[str, Any]:
    return {
        "title": task.title,
        "research_question": task.research_question,
        "keywords": task.keywords,
        "requirements": task.user_requirements,
        "review_type": task.output.review_type.value,
        "language": task.output.language,
        "reference_style": task.output.reference_style,
    }
