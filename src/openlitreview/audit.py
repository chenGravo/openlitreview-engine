from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .fulltext import license_allows_private_processing
from .schemas import EvidenceCard, SearchRun, TaskSpec
from .storage import citation_key

CITATION_RE = re.compile(r"@([A-Za-z0-9_.:-]+)")
SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
TRAILING_CITATIONS_RE = re.compile(r"(?:\s*\[@[^\]]+\])+\s*$")
SENTENCE_ENDINGS = frozenset("。！？；：.!?;:…")
TRAILING_CLOSERS = frozenset("”’\"'）)]】》〉」』*_`")
ADVERSE_PUBLICATION_STATUSES = frozenset(
    {"retracted", "withdrawn", "expression_of_concern"}
)
VERIFIED_PUBLICATION_STATUSES = frozenset({"no_adverse_update_found", "updated"})


def audit_run(
    task: TaskSpec,
    run: SearchRun,
    cards: list[EvidenceCard] | None,
    markdown: str | None,
    reviewer: dict[str, Any] | None,
    output: Path,
) -> dict[str, Any]:
    known = {citation_key(paper): paper for paper in run.papers}
    known_by_record = {paper.record_id: paper for paper in run.papers}
    operational_sources = [
        source.value
        for source in task.search.sources
        if run.source_status.get(source.value, {}).get("status") == "ok"
    ]
    contributing_sources = [
        source.value
        for source in task.search.sources
        if int(run.source_status.get(source.value, {}).get("records") or 0) > 0
    ]
    findings: list[dict[str, str]] = []
    if len(contributing_sources) < task.search.minimum_independent_sources:
        findings.append(
            {
                "severity": "high",
                "code": "insufficient_sources",
                "message": (
                    f"Only {len(contributing_sources)} independent sources contributed records; "
                    f"task requires {task.search.minimum_independent_sources}."
                ),
            }
        )
    if not run.papers:
        findings.append(
            {"severity": "high", "code": "no_papers", "message": "No papers were retrieved."}
        )
    elif len(run.papers) < task.quality.minimum_retained_records:
        findings.append(
            {
                "severity": "high",
                "code": "insufficient_retained_records",
                "message": (
                    f"Only {len(run.papers)} records were retained; task requires at least "
                    f"{task.quality.minimum_retained_records}."
                ),
            }
        )
    used_keys: set[str] = set()
    substantive_evidence_keys: set[str] = set()
    bibliographic_metadata_papers: set[str] = set()
    unknown_keys: set[str] = set()
    uncited_paragraphs: list[str] = []
    unfinished_sections: list[str] = []
    if markdown:
        used_keys = set(CITATION_RE.findall(markdown))
        unknown_keys = used_keys - set(known)
        if unknown_keys:
            findings.append(
                {
                    "severity": "high",
                    "code": "unknown_citations",
                    "message": f"Unknown citation keys: {sorted(unknown_keys)}",
                }
            )
        for paragraph in re.split(r"\n\s*\n", markdown):
            cleaned = paragraph.strip()
            if (
                len(cleaned) >= 120
                and not cleaned.startswith(("---", "#", "|", "**关键词"))
                and not CITATION_RE.search(cleaned)
            ):
                uncited_paragraphs.append(cleaned[:160])
        if uncited_paragraphs:
            findings.append(
                {
                    "severity": "medium",
                    "code": "long_paragraphs_without_citations",
                    "message": (
                        f"Found {len(uncited_paragraphs)} long paragraphs without citations."
                    ),
                }
            )
        unfinished_sections = _unfinished_section_endings(markdown)
        if unfinished_sections:
            findings.append(
                {
                    "severity": "high",
                    "code": "unfinished_section_endings",
                    "message": (
                        "Section text does not end with complete sentence punctuation: "
                        f"{unfinished_sections}"
                    ),
                }
            )
        adverse = [
            key
            for key in used_keys & set(known)
            if known[key].publication_status in ADVERSE_PUBLICATION_STATUSES
        ]
        if adverse:
            findings.append(
                {
                    "severity": "high",
                    "code": "adverse_publication_cited",
                    "message": f"Adverse publication-status records cited: {adverse}",
                }
            )
    if cards is not None and not cards:
        findings.append(
            {
                "severity": "high",
                "code": "no_evidence_cards",
                "message": "No auditable evidence cards were produced.",
            }
        )
    if cards:
        all_evidence_papers = {card.record_id for card in cards}
        evidence_papers = {
            card.record_id
            for card in cards
            if card.evidence_type != "bibliographic_metadata"
        }
        bibliographic_metadata_papers = {
            card.record_id
            for card in cards
            if card.evidence_type == "bibliographic_metadata"
        }
        substantive_evidence_keys = {
            citation_key(known_by_record[record_id])
            for record_id in evidence_papers
            if record_id in known_by_record
        }
        fulltext_papers = {card.record_id for card in cards if card.fulltext_verified}
        if len(evidence_papers) < task.quality.minimum_evidence_papers:
            findings.append(
                {
                    "severity": "high",
                    "code": "insufficient_evidence_papers",
                    "message": (
                        f"Only {len(evidence_papers)} papers produced evidence; task requires "
                        f"{task.quality.minimum_evidence_papers}."
                    ),
                }
            )
        if len(fulltext_papers) < task.quality.minimum_fulltext_verified_papers:
            findings.append(
                {
                    "severity": "high",
                    "code": "insufficient_fulltext_evidence",
                    "message": (
                        f"Only {len(fulltext_papers)} evidence papers were verified from lawful "
                        f"full text; task requires {task.quality.minimum_fulltext_verified_papers}."
                    ),
                }
            )
        publication_statuses = Counter(
            known_by_record[record_id].publication_status
            for record_id in evidence_papers
            if record_id in known_by_record
        )
        publication_checked_papers = sum(
            publication_statuses[status] for status in VERIFIED_PUBLICATION_STATUSES
        )
        required_publication_checks = min(
            task.quality.minimum_evidence_papers,
            len(evidence_papers),
        )
        if publication_checked_papers < required_publication_checks:
            findings.append(
                {
                    "severity": "high",
                    "code": "insufficient_publication_status_checks",
                    "message": (
                        f"Only {publication_checked_papers} evidence papers completed a "
                        "publication-status check without an adverse update; task requires "
                        f"at least {required_publication_checks}. Status counts: "
                        f"{dict(sorted(publication_statuses.items()))}."
                    ),
                }
            )
        incomplete_publication_checks = sum(
            count
            for status, count in publication_statuses.items()
            if status not in VERIFIED_PUBLICATION_STATUSES
            and status not in ADVERSE_PUBLICATION_STATUSES
        )
        if incomplete_publication_checks:
            findings.append(
                {
                    "severity": "medium",
                    "code": "publication_status_checks_incomplete",
                    "message": (
                        f"{incomplete_publication_checks} evidence papers have an incomplete "
                        f"publication-status check: {dict(sorted(publication_statuses.items()))}."
                    ),
                }
            )
        adverse_evidence = sorted(
            record_id
            for record_id in all_evidence_papers
            if record_id in known_by_record
            and known_by_record[record_id].publication_status
            in ADVERSE_PUBLICATION_STATUSES
        )
        if adverse_evidence:
            findings.append(
                {
                    "severity": "high",
                    "code": "adverse_publication_in_evidence",
                    "message": f"Evidence cards include adverse-status records: {adverse_evidence}",
                }
            )
        fulltext_without_verified_license = sorted(
            record_id
            for record_id in fulltext_papers
            if record_id not in known_by_record
            or not license_allows_private_processing(known_by_record[record_id])
        )
        if fulltext_without_verified_license:
            findings.append(
                {
                    "severity": "high",
                    "code": "fulltext_license_not_verified",
                    "message": (
                        "Full-text evidence was marked verified without an explicit supported "
                        f"reuse license: {fulltext_without_verified_license}"
                    ),
                }
            )
    substantive_used_keys = used_keys & substantive_evidence_keys
    if markdown and len(substantive_used_keys) < task.quality.minimum_cited_papers:
        findings.append(
            {
                "severity": "high",
                "code": "insufficient_cited_papers",
                "message": (
                    f"Draft cites {len(substantive_used_keys)} substantive evidence papers; "
                    "bibliographic-metadata-only citations do not count. Task requires at least "
                    f"{task.quality.minimum_cited_papers}."
                ),
            }
        )
    reviewer_verdict = str((reviewer or {}).get("verdict") or "not_run")
    reviewer_mode = (
        "same_model"
        if task.models.primary_model == task.models.reviewer_model
        else "independent_model"
    )
    if task.models.enabled and task.models.allow_second_model_review and reviewer_verdict != "pass":
        findings.append(
            {
                "severity": "high",
                "code": "model_review_not_passed",
                "message": f"{reviewer_mode} review verdict: {reviewer_verdict}",
            }
        )
    high_reviewer_issues = [
        issue
        for issue in (reviewer or {}).get("issues") or []
        if isinstance(issue, dict) and str(issue.get("severity") or "").lower() == "high"
    ]
    if high_reviewer_issues:
        findings.append(
            {
                "severity": "high",
                "code": "model_review_high_severity_issues",
                "message": f"Model review reported {len(high_reviewer_issues)} high issues.",
            }
        )
    status = "pass" if not any(item["severity"] == "high" for item in findings) else "blocked"
    report = {
        "status": status,
        "task_id": run.task_id,
        "operational_sources": operational_sources,
        "contributing_sources": contributing_sources,
        "successful_sources": operational_sources,
        "raw_records": run.raw_record_count,
        "deduplicated_records": run.deduplicated_record_count,
        "retained_records": len(run.papers),
        "evidence_cards": len(cards or []),
        "evidence_papers": len(
            {
                card.record_id
                for card in cards or []
                if card.evidence_type != "bibliographic_metadata"
            }
        ),
        "bibliographic_metadata_papers": len(bibliographic_metadata_papers),
        "fulltext_verified_papers": len(
            {card.record_id for card in cards or [] if card.fulltext_verified}
        ),
        "fulltext_verified_cards": sum(card.fulltext_verified for card in cards or []),
        "fulltext_license_verified_papers": len(
            {
                card.record_id
                for card in cards or []
                if card.fulltext_verified
                and card.record_id in known_by_record
                and license_allows_private_processing(known_by_record[card.record_id])
            }
        ),
        "publication_status_counts": dict(
            sorted(
                Counter(
                    known_by_record[record_id].publication_status
                    for record_id in {card.record_id for card in cards or []}
                    if record_id in known_by_record
                ).items()
            )
        ),
        "publication_status_checked_papers": len(
            {
                card.record_id
                for card in cards or []
                if card.record_id in known_by_record
                and known_by_record[card.record_id].publication_status
                in VERIFIED_PUBLICATION_STATUSES
            }
        ),
        "citations_used": len(used_keys),
        "substantive_citations_used": len(substantive_used_keys),
        "unknown_citations": sorted(unknown_keys),
        "uncited_long_paragraphs": uncited_paragraphs,
        "unfinished_sections": unfinished_sections,
        "model_review_mode": reviewer_mode,
        "model_review_verdict": reviewer_verdict,
        "findings": findings,
        "human_review_required": True,
        "review_type_claim": "narrative review; not a systematic review",
    }
    audit_dir = output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (audit_dir / "quality_report.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    return report


def _unfinished_section_endings(markdown: str) -> list[str]:
    """Find prose sections whose final substantive sentence appears truncated."""
    headings = list(SECTION_HEADING_RE.finditer(markdown))
    unfinished: list[str] = []
    for index, match in enumerate(headings):
        heading = match.group(1).strip()
        if heading in {"摘要", "参考文献"}:
            continue
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        body = markdown[match.end() : body_end].strip()
        if not body:
            unfinished.append(heading)
            continue
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", body) if item.strip()]
        if not paragraphs:
            unfinished.append(heading)
            continue
        ending = TRAILING_CITATIONS_RE.sub("", paragraphs[-1]).rstrip()
        while ending and ending[-1] in TRAILING_CLOSERS:
            ending = ending[:-1].rstrip()
        if not ending or ending[-1] not in SENTENCE_ENDINGS:
            unfinished.append(heading)
    return unfinished


def _report_markdown(report: dict[str, Any]) -> str:
    findings = report.get("findings") or []
    finding_lines = (
        "\n".join(
            f"- [{item['severity']}] `{item['code']}`：{item['message']}" for item in findings
        )
        or "- 未发现阻断性自动检查问题。"
    )
    return f"""# 质量审计报告

- 自动审计状态：**{report['status']}**
- 正常响应数据源：{len(report['operational_sources'])}
- 实际贡献记录的数据源：{len(report['contributing_sources'])}
- 原始记录：{report['raw_records']}
- 去重后记录：{report['deduplicated_records']}
- 保留记录：{report['retained_records']}
- 证据卡片：{report['evidence_cards']}
- 进入证据矩阵的文献：{report['evidence_papers']}
- 仅用于历史定位的书目元数据文献：{report['bibliographic_metadata_papers']}
- 全文核验文献：{report['fulltext_verified_papers']}
- 明确许可支持的全文文献：{report['fulltext_license_verified_papers']}
- 完成发表状态核查的证据文献：{report['publication_status_checked_papers']}
- 发表状态分布：{report['publication_status_counts']}
- 正文引用文献：{report['citations_used']}
- 正文引用的实质性证据文献：{report['substantive_citations_used']}
- 模型复核方式：{report['model_review_mode']}
- 模型复核结论：{report['model_review_verdict']}

## 发现

{finding_lines}

## 说明

这是程序审计，不替代用户的学术判断、学校或期刊的正式审查，也不替代指定查重系统。
本项目输出为普通叙述性文献综述，不声称采用系统综述方法或穷尽所有数据库。
"""
