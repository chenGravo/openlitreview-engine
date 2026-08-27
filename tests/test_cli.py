from __future__ import annotations

import json

from openlitreview.cli import main
from openlitreview.config import load_task


def test_quick_task_creates_valid_three_model_task(tmp_path) -> None:
    output = tmp_path / "quick-task.json"

    result = main(
        [
            "quick-task",
            "--title",
            "功能性训练研究综述",
            "--question",
            "功能性训练如何影响体育教育结果与医学安全？",
            "--keywords",
            "functional training；physical education, injury prevention",
            "--year-from",
            "2010",
            "--year-to",
            "2026",
            "--characters",
            "6000",
            "--requirements",
            "普通中文叙述性综述。",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    task = load_task(output)
    assert task.keywords == [
        "functional training",
        "physical education",
        "injury prevention",
    ]
    assert task.output.target_chinese_characters == 6000
    assert task.models.enabled is True
    assert task.models.primary_model == "deepseek-v4-pro"
    assert task.models.perspective_model == "kimi-k2.6"
    assert task.models.reviewer_model == "doubao-seed-2.1-pro"
    assert json.loads(output.read_text(encoding="utf-8"))["languages"] == ["en"]
