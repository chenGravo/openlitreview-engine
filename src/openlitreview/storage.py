from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .schemas import PaperRecord, SearchRun, TaskSpec


def prepare_run_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "search").mkdir(exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    (output / "draft").mkdir(exist_ok=True)
    (output / "audit").mkdir(exist_ok=True)
    (output / "private_work").mkdir(exist_ok=True)
    return output


def write_search_outputs(run: SearchRun, task: TaskSpec, output: Path) -> None:
    search_dir = output / "search"
    (search_dir / "search_run.json").write_text(
        run.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    papers = [paper.model_dump(mode="json") for paper in run.papers]
    (search_dir / "papers.json").write_text(
        json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "rank",
        "citation_key",
        "title",
        "year",
        "authors",
        "venue",
        "doi",
        "pmid",
        "citation_count",
        "rank_score",
        "sources",
        "open_access_pdf_url",
        "publication_status",
        "quality_flags",
    ]
    with (search_dir / "papers.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, paper in enumerate(run.papers, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "citation_key": citation_key(paper),
                    "title": paper.title,
                    "year": paper.year or "",
                    "authors": "; ".join(paper.authors),
                    "venue": paper.venue or "",
                    "doi": paper.doi or "",
                    "pmid": paper.pmid or "",
                    "citation_count": paper.citation_count,
                    "rank_score": paper.rank_score,
                    "sources": "; ".join(paper.source_names),
                    "open_access_pdf_url": paper.open_access_pdf_url or "",
                    "publication_status": paper.publication_status,
                    "quality_flags": "; ".join(paper.quality_flags),
                }
            )
    references = [paper_to_csl(paper) for paper in run.papers]
    (search_dir / "references.csl.json").write_text(
        json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (search_dir / "search_report.md").write_text(
        render_search_report(run, task), encoding="utf-8"
    )


def render_search_report(run: SearchRun, task: TaskSpec) -> str:
    source_rows = []
    for source in task.search.sources:
        source_name = source.value
        status = run.source_status.get(source_name, {})
        source_rows.append(
            f"| {source_name} | {status.get('status', 'unknown')} | "
            f"{status.get('records', 0)} |"
        )
    citation_chase = run.source_status.get("citation_chase") or {}
    abstract_count = sum(bool(paper.abstract) for paper in run.papers)
    oa_count = sum(bool(paper.open_access_pdf_url) for paper in run.papers)
    top_rows = []
    for index, paper in enumerate(run.papers[:30], start=1):
        title = paper.title.replace("|", "\\|")
        top_rows.append(
            f"| {index} | {paper.rank_score:.3f} | {paper.year or ''} | "
            f"{paper.citation_count} | {title} |"
        )
    return f"""# 检索报告

- 任务：{task.title}
- 任务 ID：{run.task_id}
- 检索开始：{run.started_at}
- 检索完成：{run.completed_at}
- 检索式数量：{len(run.queries)}
- 原始记录：{run.raw_record_count}
- 去重后记录：{run.deduplicated_record_count}
- 排序后保留：{len(run.papers)}
- 含摘要记录：{abstract_count}
- 提供开放 PDF 位置：{oa_count}
- 引文追踪新增记录：{citation_chase.get('records_added', 0)}
- 新增高相关文献饱和：{citation_chase.get('saturation_reached', False)}

## 数据源执行情况

| 来源 | 状态 | 返回记录 |
|---|---:|---:|
{chr(10).join(source_rows)}

## 实际检索式

{chr(10).join(f'- `{query}`' for query in run.queries)}

## 前 30 条候选

| 排名 | 综合分 | 年份 | 引用量 | 题名 |
|---:|---:|---:|---:|---|
{chr(10).join(top_rows)}

## 边界说明

本报告是普通叙述性综述的可审计检索记录，不代表系统综述，也不声称穷尽全部文献。
引用量只作为分项指标；最终纳入还需结合相关度、研究设计、全文可用性、观点覆盖和发表状态。
“饱和”仅表示本次引文追踪新增且进入筛选池的高相关记录比例低于任务阈值，不代表不存在其他文献。
"""


def citation_key(paper: PaperRecord) -> str:
    identity = paper.doi or paper.pmid or paper.arxiv_id or f"{paper.title}|{paper.year}"
    digest = hashlib.sha1(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"ref_{digest}"


def paper_to_csl(paper: PaperRecord) -> dict[str, object]:
    item: dict[str, object] = {
        "id": citation_key(paper),
        "type": _csl_type(paper.work_type),
        "title": paper.title,
        "author": [{"literal": author} for author in paper.authors],
    }
    if paper.venue:
        item["container-title"] = paper.venue
    if paper.year:
        item["issued"] = {"date-parts": [[paper.year]]}
    if paper.doi:
        item["DOI"] = paper.doi
    if paper.landing_page_url:
        item["URL"] = paper.landing_page_url
    return item


def _csl_type(work_type: str | None) -> str:
    normalized = (work_type or "").lower()
    if any(term in normalized for term in ("book", "monograph")):
        return "book"
    if "chapter" in normalized:
        return "chapter"
    if any(term in normalized for term in ("proceeding", "conference")):
        return "paper-conference"
    if any(term in normalized for term in ("thesis", "dissertation")):
        return "thesis"
    if "report" in normalized:
        return "report"
    return "article-journal"
