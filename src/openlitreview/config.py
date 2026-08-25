from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from docx import Document
from pypdf import PdfReader

from .schemas import TaskSpec, safe_resolve

MAX_REQUIREMENTS_FILE_BYTES = 10 * 1024 * 1024
SUPPORTED_REQUIREMENTS_SUFFIXES = {".md", ".txt", ".docx", ".pdf"}


def load_task(path: str | Path) -> TaskSpec:
    task_path = Path(path)
    if not task_path.is_file():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    raw = task_path.read_text(encoding="utf-8")
    if task_path.suffix.lower() == ".json":
        data: Any = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Task file must contain a mapping/object at its root")
    requirements_name = data.get("writing_requirements_file")
    if requirements_name:
        requirements_path = safe_resolve(task_path.parent, str(requirements_name))
        extracted = _read_requirements_file(requirements_path)
        inline = str(data.get("user_requirements") or "").strip()
        data["user_requirements"] = "\n\n".join(value for value in (inline, extracted) if value)
    return TaskSpec.model_validate(data)


def _read_requirements_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Writing requirements file not found: {path.name}")
    if path.suffix.lower() not in SUPPORTED_REQUIREMENTS_SUFFIXES:
        raise ValueError(
            "Writing requirements file must be Markdown, TXT, DOCX, or PDF"
        )
    if path.stat().st_size > MAX_REQUIREMENTS_FILE_BYTES:
        raise ValueError("Writing requirements file exceeds the 10 MB limit")
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".docx":
        document = Document(path)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    cleaned = text.strip()
    if len(cleaned) < 10:
        raise ValueError("Writing requirements file contains too little extractable text")
    return cleaned[:100_000]


def dump_task_contract(task: TaskSpec, path: str | Path) -> None:
    contract = {
        "task_id": task.resolved_task_id(),
        "title": task.title,
        "research_question": task.research_question,
        "keywords": task.keywords,
        "languages": task.languages,
        "year_range": [task.year_from, task.year_to],
        "base_queries": task.base_queries(),
        "sources": [source.value for source in task.search.sources],
        "candidate_target": task.search.max_candidates,
        "screening_pool": task.search.screening_pool,
        "target_fulltexts": task.search.target_fulltexts,
        "user_requirements": task.user_requirements,
        "writing_requirements_file": task.writing_requirements_file,
        "output": task.output.model_dump(mode="json"),
        "quality": task.quality.model_dump(mode="json"),
        "budget": task.budget.model_dump(mode="json"),
        "models": task.models.model_dump(mode="json"),
        "compliance": task.compliance.model_dump(mode="json"),
        "requires_human_confirmation": True,
    }
    Path(path).write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
