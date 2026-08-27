from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .network import reject_contact_identifiers


class ReviewType(StrEnum):
    NARRATIVE = "narrative"


class PrivacyClass(StrEnum):
    PUBLIC = "A"
    LAWFUL_NONPUBLIC = "B"


class SearchSourceName(StrEnum):
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    EUROPE_PMC = "europe_pmc"


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SearchSourceName] = Field(
        default_factory=lambda: [
            SearchSourceName.CROSSREF,
            SearchSourceName.SEMANTIC_SCHOLAR,
            SearchSourceName.EUROPE_PMC,
        ]
    )
    max_queries_per_source: int = Field(default=8, ge=1, le=20)
    max_candidates: int = Field(default=10_000, ge=100, le=20_000)
    max_per_query_per_source: int = Field(default=500, ge=20, le=1_000)
    screening_pool: int = Field(default=1_000, ge=50, le=2_000)
    target_fulltexts: int = Field(default=80, ge=10, le=150)
    minimum_independent_sources: int = Field(default=3, ge=2, le=8)
    saturation_rounds: int = Field(default=2, ge=1, le=5)
    saturation_new_relevant_ratio: float = Field(default=0.02, gt=0, le=0.2)
    citation_seed_count: int = Field(default=0, ge=0, le=30)
    max_citation_expansion: int = Field(default=0, ge=0, le=5_000)
    include_preprints: bool = True
    open_access_fulltext_only: bool = True


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "zh-CN"
    target_chinese_characters: int = Field(default=8_000, ge=1_500, le=50_000)
    review_type: ReviewType = ReviewType.NARRATIVE
    reference_style: str = "GB/T 7714-2025-numeric-bilingual"
    formats: list[str] = Field(default_factory=lambda: ["md", "docx", "pdf"])

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: list[str]) -> list[str]:
        allowed = {"md", "docx", "pdf"}
        cleaned = list(dict.fromkeys(item.lower() for item in value))
        unknown = set(cleaned) - allowed
        if unknown:
            raise ValueError(f"Unsupported output formats: {sorted(unknown)}")
        return cleaned


class BudgetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_reservation_cny: float = Field(default=15.0, gt=0, le=30)
    monthly_warning_cny: float = Field(default=80.0, gt=0, le=100)
    monthly_hard_stop_cny: float = Field(default=90.0, gt=0, le=100)
    external_monthly_cap_cny: float = Field(default=100.0, gt=0, le=100)
    single_request_cap_cny: float = Field(default=5.0, gt=0, le=10)
    per_model_task_cap_cny: float = Field(default=10.0, gt=0, le=10)
    monthly_per_model_cap_cny: float = Field(default=10.0, gt=0, le=10)

    @model_validator(mode="after")
    def validate_thresholds(self) -> BudgetSettings:
        if not (
            self.monthly_warning_cny
            < self.monthly_hard_stop_cny
            < self.external_monthly_cap_cny + 1e-9
        ):
            raise ValueError("Budget thresholds must satisfy warning < hard stop <= external cap")
        if self.single_request_cap_cny > self.per_model_task_cap_cny:
            raise ValueError("Single request cap must not exceed the per-model task cap")
        if self.single_request_cap_cny > self.monthly_per_model_cap_cny:
            raise ValueError("Single request cap must not exceed the monthly per-model cap")
        return self


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    cheap_model: str = "deepseek-v4-flash"
    primary_model: str = "deepseek-v4-pro"
    perspective_model: str = "kimi-k2.6"
    reviewer_model: str = "doubao-seed-2.1-pro"
    allow_second_model_review: bool = True
    max_revision_rounds: int = Field(default=1, ge=0, le=2)
    temperature: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_reviewed_models(self) -> ModelSettings:
        from .pricing import MODEL_PRICES

        aliases = (
            self.cheap_model,
            self.primary_model,
            self.perspective_model,
            self.reviewer_model,
        )
        unknown = [alias for alias in aliases if alias not in MODEL_PRICES]
        if unknown:
            raise ValueError(f"Unknown or unpriced model aliases: {unknown}")
        if (
            self.enabled
            and self.allow_second_model_review
            and self.primary_model == self.reviewer_model
        ):
            raise ValueError("Independent reviewer_model must differ from primary_model")
        if self.enabled and self.perspective_model in {
            self.primary_model,
            self.reviewer_model,
        }:
            raise ValueError(
                "Independent perspective_model must differ from primary_model and "
                "reviewer_model"
            )
        return self


class QualitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_retained_records: int = Field(default=30, ge=5, le=500)
    minimum_evidence_papers: int = Field(default=15, ge=3, le=100)
    minimum_fulltext_verified_papers: int = Field(default=5, ge=0, le=100)
    minimum_cited_papers: int = Field(default=12, ge=3, le=100)

    @model_validator(mode="after")
    def validate_quality_thresholds(self) -> QualitySettings:
        if self.minimum_fulltext_verified_papers > self.minimum_evidence_papers:
            raise ValueError(
                "minimum_fulltext_verified_papers cannot exceed minimum_evidence_papers"
            )
        if self.minimum_cited_papers > self.minimum_evidence_papers:
            raise ValueError("minimum_cited_papers cannot exceed minimum_evidence_papers")
        return self


class ComplianceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    privacy_class: PrivacyClass = PrivacyClass.PUBLIC
    rights_confirmed: bool = True
    contains_personal_or_sensitive_data: bool = False
    contains_confidential_or_secret_data: bool = False
    ai_policy_path: str | None = None
    ai_body_generation_allowed: bool | None = None
    ai_disclosure_required: bool | None = None

    @model_validator(mode="after")
    def reject_unsupported_material(self) -> ComplianceSettings:
        if self.contains_personal_or_sensitive_data:
            raise ValueError("Phase 0 does not accept personal or sensitive data")
        if self.contains_confidential_or_secret_data:
            raise ValueError("Phase 0 does not accept confidential or secret data")
        if not self.rights_confirmed:
            raise ValueError("Material rights must be confirmed before cloud processing")
        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    title: str = Field(min_length=3, max_length=300)
    research_question: str = Field(min_length=5, max_length=2_000)
    keywords: list[str] = Field(min_length=1, max_length=50)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    year_from: int | None = Field(default=None, ge=1800, le=2100)
    year_to: int | None = Field(default=None, ge=1800, le=2100)
    include_terms: list[str] = Field(default_factory=list, max_length=100)
    exclude_terms: list[str] = Field(default_factory=list, max_length=100)
    user_requirements: str = ""
    writing_requirements_file: str | None = None
    search: SearchSettings = Field(default_factory=SearchSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)

    @field_validator("keywords", "languages", "include_terms", "exclude_terms")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def validate_dates_and_scope(self) -> TaskSpec:
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must not exceed year_to")
        if not set(lang.lower() for lang in self.languages).issubset({"en", "all"}):
            raise ValueError("Phase 0 is configured for English or all-language retrieval")
        for query_part in [*self.keywords, *self.include_terms]:
            reject_contact_identifiers(query_part)
        if SearchSourceName.OPENALEX in self.search.sources:
            raise ValueError(
                "OpenAlex is unavailable in anonymous-only mode because it requires an "
                "account-linked API key"
            )
        return self

    def resolved_task_id(self) -> str:
        if self.task_id:
            return re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.task_id).strip("-")[:80]
        digest = hashlib.sha256(
            f"{self.title}|{self.research_question}|{date.today().isoformat()}".encode()
        ).hexdigest()[:10]
        return f"review-{date.today().isoformat()}-{digest}"

    def base_queries(self) -> list[str]:
        quoted = [f'"{term}"' if " " in term else term for term in self.keywords]
        primary = " AND ".join(quoted)
        queries = [primary, *self.keywords, *self.include_terms]
        normalized: list[str] = []
        for query in queries:
            query = unicodedata.normalize("NFKC", query).strip()
            if query and query not in normalized:
                normalized.append(query)
        return normalized[: self.search.max_queries_per_source]


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    record_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    publication_date: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    work_type: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    citation_count: int = 0
    influential_citation_count: int = 0
    source_names: list[str] = Field(default_factory=list)
    source_ids: dict[str, str] = Field(default_factory=dict)
    landing_page_url: str | None = None
    open_access_pdf_url: str | None = None
    open_access_license: str | None = None
    topics: list[str] = Field(default_factory=list)
    source_relevance: float = 0.0
    rank_score: float = 0.0
    rank_breakdown: dict[str, float] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    publication_status: str = "unchecked"
    publication_updates: list[dict[str, str]] = Field(default_factory=list)
    referenced_work_ids: list[str] = Field(default_factory=list)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip().lower()
        normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized)
        normalized = re.sub(r"^doi:\s*", "", normalized)
        return normalized.rstrip(" .") or None

    def canonical_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        normalized_title = normalize_title(self.title)
        year = self.year or 0
        return f"title:{normalized_title}:{year}"


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    record_id: str
    claim: str
    evidence_type: str
    study_design: str | None = None
    population: str | None = None
    result: str
    limitations: list[str] = Field(default_factory=list)
    locator: str | None = None
    fulltext_verified: bool = False
    confidence: str = "medium"


class SearchRun(BaseModel):
    task_id: str
    started_at: str
    completed_at: str | None = None
    queries: list[str]
    source_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    raw_record_count: int = 0
    deduplicated_record_count: int = 0
    papers: list[PaperRecord] = Field(default_factory=list)


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title).lower()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def safe_resolve(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    base_resolved = base.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise ValueError(f"Path escapes task workspace: {relative}")
    return candidate
