from openlitreview.fulltext import (
    assess_substantive_fulltext,
    license_allows_private_processing,
)
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


def test_pmc_identifier_does_not_replace_an_explicit_license() -> None:
    assert not license_allows_private_processing(_paper(None, pmcid="PMC123"))


def test_single_page_abstract_is_not_verified_as_fulltext() -> None:
    extracted = "[Page 1] Abstract " + ("reported finding " * 200)

    verified, reason = assess_substantive_fulltext(extracted, pages=1)

    assert verified is False
    assert reason == "single_page_document"


def test_pdf_that_merely_repeats_abstract_is_not_verified_as_fulltext() -> None:
    abstract = "A detailed abstract sentence. " * 60
    extracted = f"[Page 1] {abstract}\n\n[Page 2] {abstract[:800]}"

    verified, reason = assess_substantive_fulltext(extracted, pages=2, abstract=abstract)

    assert verified is False
    assert reason == "document_is_not_substantially_longer_than_abstract"


def test_structured_multi_page_article_is_verified_as_fulltext() -> None:
    extracted = "\n\n".join(
        [
            "[Page 1] Introduction " + ("context evidence " * 80),
            "[Page 2] Methods " + ("sampling and analysis " * 80),
            "[Page 3] Results " + ("observed finding " * 80),
        ]
    )

    verified, reason = assess_substantive_fulltext(extracted, pages=3)

    assert verified is True
    assert reason == "substantive_fulltext"
