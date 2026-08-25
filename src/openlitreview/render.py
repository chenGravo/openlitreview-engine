from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


def render_documents(output: Path) -> dict[str, Any]:
    draft = output / "draft" / "review.md"
    bibliography = output / "search" / "references.csl.json"
    report: dict[str, Any] = {"docx": "not_attempted", "pdf": "not_attempted"}
    if not draft.is_file() or not bibliography.is_file():
        report["error"] = "Draft or bibliography is missing"
        _write_report(output, report)
        return report
    pandoc = shutil.which("pandoc")
    node = shutil.which("node")
    renderer = os.getenv("OPENLITREVIEW_CITEPROC_RENDERER")
    if not pandoc or not node or not renderer or not Path(renderer).is_file():
        report["error"] = "pandoc, node, or the pinned citeproc-js renderer is unavailable"
        _write_report(output, report)
        return report
    style_resource = files("openlitreview").joinpath(
        "assets/gb-t-7714-2025-numeric-bilingual.csl"
    )
    asset_resource = files("openlitreview").joinpath("assets")
    docx = output / "draft" / "review.docx"
    cited_markdown = output / "draft" / "review.cited.md"
    try:
        with as_file(style_resource) as style, as_file(asset_resource) as asset_dir:
            citation_command = [
                node,
                renderer,
                str(draft),
                str(bibliography),
                str(style),
                str(asset_dir),
                str(cited_markdown),
            ]
            citation_result = subprocess.run(
                citation_command, check=True, capture_output=True, text=True
            )
        command = [
            pandoc,
            str(cited_markdown),
            "--standalone",
            "--output",
            str(docx),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        _postprocess_docx(docx)
        report["docx"] = "created"
        report["citation_renderer"] = "citeproc-js-2.4.63"
        if citation_result.stderr.strip():
            report["citation_warning"] = citation_result.stderr.strip()[:1000]
        if completed.stderr.strip():
            report["pandoc_warning"] = completed.stderr.strip()[:1000]
    except Exception as exc:
        report["docx"] = "failed"
        report["docx_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        _write_report(output, report)
        return report
    libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
    if libreoffice:
        try:
            with tempfile.TemporaryDirectory(prefix="openlitreview-libreoffice-") as profile:
                subprocess.run(
                    [
                        libreoffice,
                        f"-env:UserInstallation={Path(profile).as_uri()}",
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        str(docx.parent),
                        str(docx),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            report["pdf"] = "created" if docx.with_suffix(".pdf").is_file() else "failed"
        except Exception as exc:
            report["pdf"] = "failed"
            report["pdf_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    else:
        report["pdf"] = "skipped_libreoffice_missing"
    if report["docx"] == "created":
        report["submission"] = _prepare_submission(
            output, cited_markdown, docx, docx.with_suffix(".pdf")
        )
    _write_report(output, report)
    return report


def _write_report(output: Path, report: dict[str, Any]) -> None:
    (output / "audit" / "render_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _prepare_submission(
    output: Path, cited_markdown: Path, docx: Path, pdf: Path
) -> list[str]:
    submission = output / "submission"
    submission.mkdir(exist_ok=True)
    created: list[str] = []
    for source, target_name in (
        (cited_markdown, "review.md"),
        (docx, "review.docx"),
        (pdf, "review.pdf"),
    ):
        if source.is_file():
            shutil.copy2(source, submission / target_name)
            created.append(target_name)
    return created


def _postprocess_docx(path: Path) -> None:
    document = Document(path)
    section = document.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.first_line_indent = Cm(0.74)
    normal.paragraph_format.space_after = Pt(0)

    for style_name, chinese_font, size in (
        ("Title", "黑体", 18),
        ("Heading 1", "黑体", 15),
        ("Heading 2", "黑体", 14),
        ("Heading 3", "黑体", 12),
    ):
        if style_name not in document.styles:
            continue
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), chinese_font)

    in_references = False
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == "参考文献":
            in_references = True
            continue
        if in_references and paragraph.text.strip():
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.left_indent = Cm(0.74)
            paragraph.paragraph_format.first_line_indent = Cm(-0.74)
    document.save(path)
