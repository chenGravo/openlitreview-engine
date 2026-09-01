from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .prompts import WRITING_SYSTEM
from .schemas import EvidenceCard, PaperRecord, TaskSpec
from .storage import citation_key

CITATION_CLUSTER_RE = re.compile(
    r"\[((?:@[A-Za-z0-9_.:-]+)(?:;\s*@[A-Za-z0-9_.:-]+)+)\]"
)

OUTLINE_SYSTEM = """You design a Chinese narrative literature-review outline from an audited
evidence set. Cover foundational concepts, major evidence clusters, disagreements, limitations,
and future directions. Do not invent citations or facts. Return strict JSON only."""

REVIEW_SYSTEM = """You are an academic evidence auditor. Compare the supplied
Chinese draft against the evidence cards. Identify unsupported, overstated, causal, numerical,
medical-safety, contradictory, or missing-counterevidence claims. Do not rewrite for style and do
not invent sources. Use verdict `revise` or `reject` only when at least one high-severity issue
remains; use verdict `pass` when no high-severity issue remains, even if you report medium or low
items for later human review. Return strict JSON only."""

PERSPECTIVE_SYSTEM = """You are the pre-writing cross-evidence perspective auditor for a Chinese
academic literature review. Use only the supplied evidence cards. Identify competing explanations,
contradictory or null findings, population and subgroup boundaries, education-versus-medical
boundaries, safety uncertainty, missing perspectives, and limitations that the outline and draft
must preserve. Do not invent sources, facts, identifiers, or certainty. Return strict JSON only."""

REVISION_SYSTEM = """You revise a Chinese narrative academic literature review after an
evidence audit. Correct every reported issue using only the supplied evidence cards.
Preserve supported nuance, disagreements, null results, limitations, and Pandoc citation keys.
Never invent facts or citations. Return the complete revised review as strict JSON only."""

DIGEST_SYSTEM = """You create a loss-minimizing evidence digest for one batch of an audited
academic evidence set. Produce exactly one source summary for every supplied citation key. Retain
quantitative results, null findings, disagreements, population and intervention boundaries,
study-design strength, medical-safety uncertainty, and limitations. Use only supplied evidence
IDs and citation keys. Do not infer missing results or invent facts, sources, identifiers, or
certainty. Return strict JSON only."""

SECTION_WRITING_SYSTEM = (
    WRITING_SYSTEM
    + """
Write only the requested review part. Respect its target length and supplied outline role.
For citations, use only values from the `citation_key` fields, formatted as Pandoc citations.
Never cite an `evidence_id`; evidence IDs only identify supporting snippets.
Do not add a reference list or discuss any other section. Return strict JSON only."""
)

SECTION_REVISION_SYSTEM = """You revise only one requested part of a Chinese narrative academic
literature review after an evidence audit. Correct relevant issues using only the supplied
evidence digest. Preserve supported nuance, disagreements, null results, limitations, and Pandoc
citation keys. For citations, use only values from the `citation_key` fields; never cite an
`evidence_id`. Do not invent facts or citations, and do not discuss the generation process.
Return strict JSON only."""


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
    writing_checkpoints = dict(initial_writing_checkpoints or {})
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
    writing_checkpoints["perspective_audit"] = perspective_payload
    writing_checkpoints["outline"] = outline_payload
    _write_writing_checkpoints(output, writing_checkpoints)
    draft_payload = await _generate_sectioned_draft(
        task,
        outline_payload,
        evidence_digest,
        perspective_payload,
        client,
        output,
        writing_checkpoints,
        checkpoint_key="initial",
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
            draft_payload = await _generate_sectioned_draft(
                task,
                outline_payload,
                evidence_digest,
                perspective_payload,
                client,
                output,
                writing_checkpoints,
                checkpoint_key=f"revision_{review_round + 1}",
                current_payload=draft_payload,
                reviewer_payload=reviewer_payload,
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


async def _generate_sectioned_draft(
    task: TaskSpec,
    outline_payload: dict[str, Any],
    evidence_digest: list[dict[str, Any]],
    perspective_payload: dict[str, Any],
    client: LLMClient,
    output: Path,
    writing_checkpoints: dict[str, Any],
    *,
    checkpoint_key: str,
    current_payload: dict[str, Any] | None = None,
    reviewer_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate or revise bounded review parts, then assemble them deterministically."""
    outline_sections = [
        dict(item) for item in outline_payload.get("sections") or [] if isinstance(item, dict)
    ]
    introduction_index = 0 if outline_sections else None
    conclusion_index = len(outline_sections) - 1 if len(outline_sections) >= 2 else None
    limitations_index = next(
        (
            index
            for index, item in enumerate(outline_sections)
            if "局限" in str(item.get("heading") or "")
        ),
        None,
    )
    excluded = {
        index
        for index in (introduction_index, conclusion_index, limitations_index)
        if index is not None
    }
    main_indices = [index for index in range(len(outline_sections)) if index not in excluded]
    total_target = task.output.target_chinese_characters
    structural_target = max(1_500, int(total_target * 0.30))
    section_target = max(400, int(total_target * 0.65 / max(len(main_indices), 1)))
    citation_aliases = _citation_aliases_from_digest(evidence_digest)

    all_parts = writing_checkpoints.setdefault("draft_parts", {})
    if not isinstance(all_parts, dict):
        all_parts = {}
        writing_checkpoints["draft_parts"] = all_parts
    seeded_parts = all_parts.get(checkpoint_key)
    parts = dict(seeded_parts) if isinstance(seeded_parts, dict) else {}
    all_parts[checkpoint_key] = parts

    structural = parts.get("structural")
    if not isinstance(structural, dict):
        structural_roles = [
            outline_sections[index]
            for index in (introduction_index, limitations_index, conclusion_index)
            if index is not None
        ]
        structural_ids = {
            evidence_id for role in structural_roles for evidence_id in _outline_evidence_ids(role)
        }
        current_structural = _current_structural_part(current_payload)
        structural_review = _review_payload_for_part(reviewer_payload, structural=True)
        if (
            reviewer_payload is not None
            and current_structural is not None
            and structural_review is None
        ):
            structural = current_structural
        else:
            structural = await client.complete_json(
                model_alias=task.models.primary_model,
                system=(
                    SECTION_REVISION_SYSTEM
                    if reviewer_payload is not None
                    else SECTION_WRITING_SYSTEM
                ),
                prompt=(
                    "Return schema: "
                    '{"title":"","abstract":"","keywords":[],"introduction":"",'
                    '"conclusion":"","limitations":""}. '
                    f"The combined target is about {structural_target} Chinese characters. "
                    "The abstract must not introduce facts absent from the cited review.\n"
                    + json.dumps(
                        {
                            "task": _task_payload(task),
                            "central_argument": outline_payload.get("central_argument"),
                            "outline_roles": structural_roles,
                            "evidence_digest": _filter_evidence_digest(
                                evidence_digest, structural_ids
                            ),
                            "perspective_audit": perspective_payload,
                            "current_part": current_structural,
                            "review_audit": structural_review,
                        },
                        ensure_ascii=False,
                    )
                ),
                max_output_tokens=6_000,
                temperature=0.1 if reviewer_payload is not None else task.models.temperature,
            )
        structural = _normalize_part_citations(structural, citation_aliases)
        parts["structural"] = structural
        _write_writing_checkpoints(output, writing_checkpoints)
    else:
        structural = _normalize_part_citations(structural, citation_aliases)
        parts["structural"] = structural
        _write_writing_checkpoints(output, writing_checkpoints)

    seeded_sections = parts.get("sections")
    section_parts = dict(seeded_sections) if isinstance(seeded_sections, dict) else {}
    parts["sections"] = section_parts
    assembled_sections: list[dict[str, str]] = []
    for index in main_indices:
        role = outline_sections[index]
        part_key = str(index + 1)
        section_payload = section_parts.get(part_key)
        if not isinstance(section_payload, dict):
            evidence_ids = set(_outline_evidence_ids(role))
            heading = str(role.get("heading") or "")
            current_section = _current_section_part(current_payload, heading)
            section_review = _review_payload_for_part(
                reviewer_payload,
                heading=heading,
                part_number=index + 1,
            )
            if (
                reviewer_payload is not None
                and current_section is not None
                and section_review is None
            ):
                section_payload = current_section
            else:
                section_payload = await client.complete_json(
                    model_alias=task.models.primary_model,
                    system=(
                        SECTION_REVISION_SYSTEM
                        if reviewer_payload is not None
                        else SECTION_WRITING_SYSTEM
                    ),
                    prompt=(
                        'Return schema: {"heading":"","body":""}. '
                        f"Target about {section_target} Chinese characters. Use the supplied "
                        "Pandoc citation keys for every material claim.\n"
                        + json.dumps(
                            {
                                "task": _task_payload(task),
                                "central_argument": outline_payload.get("central_argument"),
                                "section_role": role,
                                "required_disagreements": outline_payload.get(
                                    "required_disagreements"
                                ),
                                "evidence_digest": _filter_evidence_digest(
                                    evidence_digest, evidence_ids
                                ),
                                "current_part": current_section,
                                "review_audit": section_review,
                            },
                            ensure_ascii=False,
                        )
                    ),
                    max_output_tokens=3_000,
                    temperature=(
                        0.1 if reviewer_payload is not None else task.models.temperature
                    ),
                )
            section_payload = _normalize_part_citations(section_payload, citation_aliases)
            section_parts[part_key] = section_payload
            _write_writing_checkpoints(output, writing_checkpoints)
        else:
            section_payload = _normalize_part_citations(section_payload, citation_aliases)
            section_parts[part_key] = section_payload
            _write_writing_checkpoints(output, writing_checkpoints)
        body = str(section_payload.get("body") or "").strip()
        heading = str(role.get("heading") or section_payload.get("heading") or "").strip()
        if heading and body:
            assembled_sections.append({"heading": heading, "body": body})

    if not outline_sections and isinstance(structural.get("sections"), list):
        assembled_sections = [
            {"heading": str(item.get("heading") or ""), "body": str(item.get("body") or "")}
            for item in structural.get("sections") or []
            if isinstance(item, dict)
        ]
    return {
        "title": str(structural.get("title") or task.title).strip(),
        "abstract": str(structural.get("abstract") or "").strip(),
        "keywords": _string_list(structural.get("keywords")),
        "introduction": str(structural.get("introduction") or "").strip(),
        "sections": assembled_sections,
        "conclusion": str(structural.get("conclusion") or "").strip(),
        "limitations": str(structural.get("limitations") or "").strip(),
    }


def _outline_evidence_ids(role: dict[str, Any]) -> list[str]:
    return _string_list(role.get("evidence_ids"))


def _citation_aliases_from_digest(
    evidence_digest: list[dict[str, Any]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for batch in evidence_digest:
        for summary in batch.get("source_summaries") or []:
            if not isinstance(summary, dict):
                continue
            citation = str(summary.get("citation_key") or "").strip()
            if not citation:
                continue
            for evidence_id in _string_list(summary.get("evidence_ids")):
                aliases[evidence_id] = citation
    return aliases


def _normalize_part_citations(value: Any, aliases: dict[str, str]) -> Any:
    """Replace evidence-card citations with their verified paper citation keys."""
    if isinstance(value, dict):
        return {key: _normalize_part_citations(item, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_part_citations(item, aliases) for item in value]
    if not isinstance(value, str):
        return value
    normalized = value
    for evidence_id in sorted(aliases, key=len, reverse=True):
        normalized = normalized.replace(f"@{evidence_id}", f"@{aliases[evidence_id]}")
    for citation in sorted(set(aliases.values()), key=len, reverse=True):
        normalized = re.sub(
            rf"@{re.escape(citation)}_e\d+\b",
            f"@{citation}",
            normalized,
        )
    return CITATION_CLUSTER_RE.sub(_deduplicate_citation_cluster, normalized)


def _deduplicate_citation_cluster(match: re.Match[str]) -> str:
    citations: list[str] = []
    for citation in (item.strip() for item in match.group(1).split(";")):
        if citation not in citations:
            citations.append(citation)
    return "[" + "; ".join(citations) + "]"


def _filter_evidence_digest(
    evidence_digest: list[dict[str, Any]], evidence_ids: set[str]
) -> list[dict[str, Any]]:
    if not evidence_ids:
        return evidence_digest
    filtered: list[dict[str, Any]] = []
    for batch in evidence_digest:
        summaries = [
            summary
            for summary in batch.get("source_summaries") or []
            if isinstance(summary, dict)
            and evidence_ids.intersection(_string_list(summary.get("evidence_ids")))
        ]
        observations = [
            observation
            for observation in batch.get("cross_source_observations") or []
            if isinstance(observation, dict)
            and evidence_ids.intersection(_string_list(observation.get("evidence_ids")))
        ]
        if summaries or observations:
            filtered.append(
                {
                    "batch_number": batch.get("batch_number"),
                    "source_summaries": summaries,
                    "cross_source_observations": observations,
                }
            )
    return filtered or evidence_digest


def _current_structural_part(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        key: payload.get(key)
        for key in ("title", "abstract", "keywords", "introduction", "conclusion", "limitations")
    }


def _current_section_part(payload: dict[str, Any] | None, heading: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for section in payload.get("sections") or []:
        if isinstance(section, dict) and str(section.get("heading") or "") == heading:
            return section
    return None


def _review_payload_for_part(
    reviewer_payload: dict[str, Any] | None,
    *,
    heading: str = "",
    part_number: int | None = None,
    structural: bool = False,
) -> dict[str, Any] | None:
    """Return only review issues that require changing this bounded draft part."""
    if not isinstance(reviewer_payload, dict):
        return None
    issues = [
        issue for issue in reviewer_payload.get("issues") or [] if isinstance(issue, dict)
    ]
    verdict = str(reviewer_payload.get("verdict") or "").lower()
    if not issues:
        return reviewer_payload if verdict in {"revise", "reject"} else None

    relevant: list[dict[str, Any]] = []
    heading_core = re.sub(r"^\s*\d+[.、：:\s]*", "", heading).strip()
    global_markers = ("全文", "全篇", "整篇", "各节", "多处")
    structural_markers = ("标题", "摘要", "引言", "结论", "局限")
    for issue in issues:
        location = str(issue.get("location") or "").strip()
        applies_globally = not location or any(marker in location for marker in global_markers)
        if applies_globally:
            relevant.append(issue)
            continue
        if structural and any(marker in location for marker in structural_markers):
            relevant.append(issue)
            continue
        if structural:
            continue
        if part_number is not None and re.search(rf"第\s*{part_number}\s*节", location):
            relevant.append(issue)
            continue
        if heading_core and len(heading_core) >= 4 and heading_core in location:
            relevant.append(issue)

    if not relevant:
        return None
    filtered = dict(reviewer_payload)
    filtered["issues"] = relevant
    return filtered


def _write_writing_checkpoints(output: Path, checkpoints: dict[str, Any]) -> None:
    (output / "audit" / "writing_checkpoints.json").write_text(
        json.dumps(checkpoints, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
