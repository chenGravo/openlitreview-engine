from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from .network import ANONYMOUS_USER_AGENT
from .schemas import PaperRecord
from .storage import citation_key


@dataclass
class FullTextResult:
    record_id: str
    citation_key: str
    status: str
    source_url: str | None
    license: str | None
    sha256: str | None = None
    pages: int = 0
    characters: int = 0
    error: str | None = None
    text_path: str | None = None
    content_scope: str = "not_assessed"


def license_allows_private_processing(paper: PaperRecord) -> bool:
    value = (paper.open_access_license or "").lower()
    markers = (
        "creativecommons.org/licenses/",
        "creativecommons.org/publicdomain/",
        "cc by",
        "cc-by",
        "cc0",
        "creative commons",
        "public domain",
    )
    return bool(paper.open_access_pdf_url and any(marker in value for marker in markers))


_PAGE_MARKER_RE = re.compile(r"\[Page\s+\d+\]\s*", re.IGNORECASE)
_SECTION_SIGNAL_PATTERNS = (
    re.compile(r"\b(?:introduction|background|literature review)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:methods?|methodology|materials and methods|research design|"
        r"data collection|data analysis)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:results?|findings?)\b", re.IGNORECASE),
    re.compile(r"\b(?:discussion|conclusions?|implications?)\b", re.IGNORECASE),
    re.compile(r"\b(?:references|bibliography)\b", re.IGNORECASE),
)


def assess_substantive_fulltext(
    extracted: str,
    pages: int,
    abstract: str | None = None,
) -> tuple[bool, str]:
    """Conservatively distinguish a substantive article from an abstract or excerpt.

    Downloading a PDF proves only the file format, not that the file contains the full
    article.  The quality label therefore requires a minimum extent plus either multiple
    article-section signals or enough pages and text to make an abstract-only document
    implausible.  Rejected documents can still fall back to abstract-level evidence.
    """

    normalized = re.sub(r"\s+", " ", _PAGE_MARKER_RE.sub(" ", extracted)).strip()
    character_count = len(normalized)
    if pages < 2:
        return False, "single_page_document"
    if character_count < 2_000:
        return False, "insufficient_extracted_text"

    normalized_abstract = re.sub(r"\s+", " ", abstract or "").strip()
    if len(normalized_abstract) >= 200 and character_count < max(
        2_000, int(len(normalized_abstract) * 1.75)
    ):
        return False, "document_is_not_substantially_longer_than_abstract"

    section_signals = sum(bool(pattern.search(normalized)) for pattern in _SECTION_SIGNAL_PATTERNS)
    if section_signals < 2 and not (pages >= 4 and character_count >= 6_000):
        return False, "insufficient_fulltext_structure"
    return True, "substantive_fulltext"


async def collect_fulltexts(
    papers: list[PaperRecord],
    work_dir: Path,
    target: int,
    *,
    max_bytes: int = 25 * 1024 * 1024,
) -> list[FullTextResult]:
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[FullTextResult] = []
    accepted = 0
    async with httpx.AsyncClient(
        headers={"User-Agent": ANONYMOUS_USER_AGENT},
        timeout=httpx.Timeout(90.0, connect=20.0),
        follow_redirects=True,
    ) as client:
        for paper in papers:
            if accepted >= target:
                break
            key = citation_key(paper)
            if not license_allows_private_processing(paper):
                results.append(
                    FullTextResult(
                        record_id=paper.record_id,
                        citation_key=key,
                        status="skipped_license_or_oa_unconfirmed",
                        source_url=paper.open_access_pdf_url,
                        license=paper.open_access_license,
                    )
                )
                continue
            result = await _download_and_extract(client, paper, work_dir, max_bytes)
            results.append(result)
            if result.status == "extracted":
                accepted += 1
    (work_dir / "fulltext_manifest.json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


async def _download_and_extract(
    client: httpx.AsyncClient,
    paper: PaperRecord,
    work_dir: Path,
    max_bytes: int,
) -> FullTextResult:
    url = paper.open_access_pdf_url
    key = citation_key(paper)
    parsed = urlparse(url or "")
    if parsed.scheme != "https" or not parsed.netloc:
        return FullTextResult(
            record_id=paper.record_id,
            citation_key=key,
            status="rejected_url",
            source_url=url,
            license=paper.open_access_license,
        )
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > max_bytes:
                raise ValueError("PDF exceeds maximum allowed size")
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("PDF exceeds maximum allowed size")
                chunks.append(chunk)
        content = b"".join(chunks)
        if not content.startswith(b"%PDF"):
            raise ValueError("Downloaded content is not a PDF")
        reader = PdfReader(BytesIO(content))
        page_texts: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                page_texts.append(f"[Page {index}] {text}")
        extracted = "\n\n".join(page_texts)
        verified, verification_reason = assess_substantive_fulltext(
            extracted,
            len(reader.pages),
            paper.abstract,
        )
        if not verified:
            return FullTextResult(
                record_id=paper.record_id,
                citation_key=key,
                status="rejected_non_substantive_fulltext",
                source_url=url,
                license=paper.open_access_license,
                sha256=hashlib.sha256(content).hexdigest(),
                pages=len(reader.pages),
                characters=len(extracted),
                error=verification_reason,
                content_scope="abstract_or_excerpt",
            )
        text_path = work_dir / f"{key}.txt"
        text_path.write_text(extracted, encoding="utf-8")
        return FullTextResult(
            record_id=paper.record_id,
            citation_key=key,
            status="extracted",
            source_url=url,
            license=paper.open_access_license,
            sha256=hashlib.sha256(content).hexdigest(),
            pages=len(reader.pages),
            characters=len(extracted),
            text_path=str(text_path),
            content_scope="substantive_fulltext",
        )
    except Exception as exc:
        return FullTextResult(
            record_id=paper.record_id,
            citation_key=key,
            status="failed",
            source_url=url,
            license=paper.open_access_license,
            error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )
