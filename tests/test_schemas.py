from pathlib import Path

import pytest

from openlitreview.config import load_task
from openlitreview.schemas import TaskSpec


def test_example_task_is_valid() -> None:
    task = load_task(Path("examples/task.example.yml"))
    assert task.languages == ["en"]
    assert task.search.max_candidates == 10_000
    assert task.base_queries()[0].startswith('"first English concept"')


def test_invalid_budget_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="warning < hard stop"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["test"],
                "budget": {
                    "monthly_warning_cny": 95,
                    "monthly_hard_stop_cny": 90,
                    "external_monthly_cap_cny": 100,
                },
            }
        )


def test_single_request_cap_cannot_exceed_per_model_cap() -> None:
    with pytest.raises(ValueError, match="per-model task cap"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["test"],
                "budget": {
                    "single_request_cap_cny": 6,
                    "per_model_task_cap_cny": 5,
                },
            }
        )


def test_sensitive_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="personal or sensitive"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["test"],
                "compliance": {"contains_personal_or_sensitive_data": True},
            }
        )


@pytest.mark.parametrize(
    "private_term",
    ["author@example.com", "13800138000", "11010519491231002X"],
)
def test_contact_identifiers_are_rejected_from_search_terms(private_term: str) -> None:
    with pytest.raises(ValueError, match="must not contain"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["example intervention", private_term],
            }
        )


def test_openalex_is_rejected_in_anonymous_only_mode() -> None:
    with pytest.raises(ValueError, match="anonymous-only mode"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["example intervention"],
                "search": {"sources": ["openalex", "crossref"]},
            }
        )


def test_writing_requirements_file_is_loaded(tmp_path) -> None:
    (tmp_path / "requirements.md").write_text(
        "正文采用中文学术表达，并设置摘要和关键词。", encoding="utf-8"
    )
    task_file = tmp_path / "task.yml"
    task_file.write_text(
        """title: Valid title
research_question: A sufficiently long question?
keywords: [test]
user_requirements: 普通文献综述。
writing_requirements_file: requirements.md
""",
        encoding="utf-8",
    )
    task = load_task(task_file)
    assert "普通文献综述" in task.user_requirements
    assert "设置摘要和关键词" in task.user_requirements


def test_writing_requirements_file_cannot_escape_task_directory(tmp_path) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (tmp_path / "outside.md").write_text("这是一份不允许读取的外部要求文件。", encoding="utf-8")
    task_file = task_dir / "task.yml"
    task_file.write_text(
        """title: Valid title
research_question: A sufficiently long question?
keywords: [test]
writing_requirements_file: ../outside.md
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes task workspace"):
        load_task(task_file)


def test_unknown_model_alias_is_rejected_during_task_validation() -> None:
    with pytest.raises(ValueError, match="Unknown or unpriced"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["test"],
                "models": {"primary_model": "unreviewed-model"},
            }
        )


def test_independent_reviewer_must_use_a_different_model() -> None:
    with pytest.raises(ValueError, match="must differ"):
        TaskSpec.model_validate(
            {
                "title": "Valid title",
                "research_question": "A sufficiently long question?",
                "keywords": ["test"],
                "models": {
                    "enabled": True,
                    "primary_model": "deepseek-v4-pro",
                    "reviewer_model": "deepseek-v4-pro",
                },
            }
        )
