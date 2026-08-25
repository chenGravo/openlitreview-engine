from __future__ import annotations

import json

from .schemas import PaperRecord, TaskSpec

QUERY_SYSTEM = """You are a scholarly-search query planner. Return strict JSON only.
Do not invent papers, authors, journals, identifiers, results, or database syntax.
Generate English literature-search concepts for broad but relevant academic retrieval."""


def query_expansion_prompt(task: TaskSpec) -> str:
    payload = {
        "title": task.title,
        "research_question": task.research_question,
        "keywords": task.keywords,
        "include_terms": task.include_terms,
        "exclude_terms": task.exclude_terms,
        "year_range": [task.year_from, task.year_to],
        "maximum_queries": task.search.max_queries_per_source,
    }
    return (
        "Create complementary search queries covering exact terminology, synonyms, "
        "broader concepts, narrower concepts, disciplinary variants, and likely null or "
        "contrary findings. Keep each query concise and in English. Return this schema: "
        '{"queries":[{"query":"...","purpose":"..."}],"concepts":{"population":[],"intervention_or_phenomenon":[],"outcomes":[],"methods":[]}}\n'
        + json.dumps(payload, ensure_ascii=False)
    )


EVIDENCE_SYSTEM = """You extract auditable evidence from scholarly text.
Use only the supplied document text and metadata. Do not infer missing details.
If the text is an abstract, never claim that methods, dosage, sample, or exact results were
verified from full text. Return strict JSON only."""


def evidence_prompt(paper: PaperRecord, text: str, fulltext_verified: bool) -> str:
    metadata = {
        "record_id": paper.record_id,
        "title": paper.title,
        "year": paper.year,
        "authors": paper.authors,
        "doi": paper.doi,
        "fulltext_verified": fulltext_verified,
    }
    schema = {
        "study_design": None,
        "population": None,
        "methods": None,
        "evidence": [
            {
                "claim": "",
                "result": "",
                "locator": "",
                "limitations": [],
                "confidence": "low|medium|high",
            }
        ],
        "paper_limitations": [],
        "relevance": "low|medium|high",
    }
    return (
        "Metadata:\n"
        + json.dumps(metadata, ensure_ascii=False)
        + "\nRequired schema:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\nDocument text:\n"
        + text
    )


WRITING_SYSTEM = """You write a Chinese narrative academic literature review from an
audited evidence set. Every material claim must cite one or more supplied citation keys in
Pandoc form, such as [@paper_key]. Do not create references, facts, samples, statistics,
causal claims, or safety advice. Present disagreements, null findings, limitations, and
evidence strength. Use natural formal Chinese without mentioning AI or the generation process.
Return strict JSON only."""
