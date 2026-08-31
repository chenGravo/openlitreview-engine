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


async def generate_review(
    task: TaskSpec,
    papers: list[PaperRecord],
    cards: list[EvidenceCard],
    client: LLMClient,
    output: Path,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    evidence_packet = _evidence_packet(papers, cards)
    perspective_payload = await _audit_perspectives(task, evidence_packet, client)
    perspective_path = output / "audit" / "prewriting_perspective_audit.json"
    perspective_path.write_text(
        json.dumps(perspective_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outline_payload = await client.complete_json(
        model_alias=task.models.primary_model,
        system=OUTLINE_SYSTEM,
        prompt=(
            "Return schema: "
            '{"central_argument":"","sections":[{"heading":"","purpose":"","evidence_ids":[]}],'
            '"required_disagreements":[],"limitations_to_state":[]}\n'
            + json.dumps(
                {
                    "task": _task_payload(task),
                    "evidence": evidence_packet,
                    "perspective_audit": perspective_payload,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=4_000,
        temperature=0.1,
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
                    "evidence": evidence_packet,
                    "perspective_audit": perspective_payload,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=min(20_000, max(6_000, task.output.target_chinese_characters * 2)),
        temperature=task.models.temperature,
    )
    markdown = render_review_markdown(draft_payload)
    draft_dir = output / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    (draft_dir / "outline.json").write_text(
        json.dumps(outline_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
            reviewer_payload = await _review_draft(
                task, evidence_packet, markdown, client
            )
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
                            "evidence": evidence_packet,
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
    evidence_packet: list[dict[str, Any]],
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
                    "evidence": evidence_packet,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=3_000,
        temperature=0.2,
    )


async def _review_draft(
    task: TaskSpec,
    evidence_packet: list[dict[str, Any]],
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
                    "evidence": evidence_packet,
                    "draft_markdown": markdown,
                },
                ensure_ascii=False,
            )
        ),
        max_output_tokens=5_000,
        temperature=0.0,
    )


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


def _evidence_packet(
    papers: list[PaperRecord], cards: list[EvidenceCard]
) -> list[dict[str, Any]]:
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
