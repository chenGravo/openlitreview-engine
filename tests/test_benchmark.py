from openlitreview.benchmark import load_suite, score_case


def test_benchmark_suite_has_twelve_cases() -> None:
    cases = load_suite("benchmarks/academic_zh_v1.json")
    assert len(cases) == 12


def test_invalid_citation_and_overclaim_lose_points() -> None:
    case = load_suite("benchmarks/academic_zh_v1.json")[0]
    response = {
        "text": "干预效果已被证实。",
        "citations": ["E1", "invented"],
    }
    score = score_case(case, response)
    assert score["total"] < 50
    assert score["invalid_citations"] == ["invented"]
    assert score["forbidden_hits"]
