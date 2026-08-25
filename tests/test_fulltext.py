from openlitreview.fulltext import license_allows_private_processing
from openlitreview.schemas import PaperRecord


def _paper(license_name: str | None, *, pmcid: str | None = None) -> PaperRecord:
    return PaperRecord(
        record_id="p1",
        title="Test",
        pmcid=pmcid,
        open_access_pdf_url="https://example.org/test.pdf",
        open_access_license=license_name,
    )


def test_explicit_creative_commons_license_is_accepted() -> None:
    assert license_allows_private_processing(
        _paper("https://creativecommons.org/licenses/by/4.0/")
    )


def test_oa_route_without_reuse_license_is_not_treated_as_a_license() -> None:
    assert not license_allows_private_processing(_paper("green"))
    assert not license_allows_private_processing(_paper("free"))
    assert not license_allows_private_processing(_paper(None))


def test_pmc_open_fulltext_is_accepted() -> None:
    assert license_allows_private_processing(_paper(None, pmcid="PMC123"))
