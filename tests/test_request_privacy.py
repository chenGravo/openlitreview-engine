from __future__ import annotations

import httpx
import pytest

import openlitreview.integrity as integrity_module
from openlitreview.schemas import PaperRecord, TaskSpec
from openlitreview.sources.crossref import CrossrefSource
from openlitreview.sources.openalex import OpenAlexSource
from openlitreview.sources.semantic_scholar import SemanticScholarSource


def _task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "title": "Example intervention review",
            "research_question": "What effects does the example intervention have?",
            "keywords": ["example intervention"],
        }
    )


@pytest.mark.asyncio
async def test_default_scholarly_headers_are_generic(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTOR", "private-github-owner")
    monkeypatch.setenv("GITHUB_REPOSITORY", "private-owner/private-repository")
    source = CrossrefSource()
    try:
        serialized = "\n".join(
            f"{key}: {value}" for key, value in source.client.headers.items()
        ).lower()
    finally:
        await source.close()

    assert "user-agent: openlitreview/0.1" in serialized
    assert "@" not in serialized
    assert "private-github-owner" not in serialized
    assert "private-repository" not in serialized


@pytest.mark.asyncio
async def test_crossref_ignores_contact_environment_variable(monkeypatch) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"message": {"items": []}})

    monkeypatch.setenv("OPENLITREVIEW_CONTACT_EMAIL", "private.owner@example.com")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": "OpenLitReview/0.1"},
    )
    source = CrossrefSource(client)
    try:
        await source.search("example intervention", _task(), 20)
    finally:
        await client.aclose()

    serialized = "\n".join(
        [str(captured[0].url), *(f"{key}: {value}" for key, value in captured[0].headers.items())]
    ).lower()
    assert "private.owner@example.com" not in serialized
    assert "mailto" not in serialized


@pytest.mark.asyncio
async def test_semantic_scholar_ignores_account_api_key(monkeypatch) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": []})

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "account-linked-private-key")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": "OpenLitReview/0.1"},
    )
    source = SemanticScholarSource(client)
    try:
        await source.search("example intervention", _task(), 20)
    finally:
        await client.aclose()

    assert captured
    assert "x-api-key" not in captured[0].headers
    assert "account-linked-private-key" not in str(captured[0].url)


@pytest.mark.asyncio
async def test_openalex_makes_no_request_even_if_key_exists(monkeypatch) -> None:
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected OpenAlex request: {request.url}")

    monkeypatch.setenv("OPENALEX_API_KEY", "account-linked-private-key")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fail_if_called))
    source = OpenAlexSource(client)
    try:
        with pytest.raises(RuntimeError, match="disabled in anonymous-only mode"):
            await source.search("example intervention", _task(), 20)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_publication_update_check_sends_no_contact_fields(monkeypatch) -> None:
    captured_init: dict[str, object] = {}
    captured_requests: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {}}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured_init.update(kwargs)

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, **kwargs: object) -> FakeResponse:
            captured_requests.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setenv("OPENLITREVIEW_CONTACT_EMAIL", "private.owner@example.com")
    monkeypatch.setattr(integrity_module.httpx, "AsyncClient", FakeClient)
    paper = PaperRecord(
        record_id="paper-1",
        title="Example intervention",
        doi="10.1234/example",
    )

    result = await integrity_module.check_publication_updates([paper])

    assert result[0].publication_status == "no_adverse_update_found"
    assert captured_requests == [
        ("https://api.crossref.org/works/10.1234%2Fexample", {})
    ]
    serialized_headers = str(captured_init.get("headers", {})).lower()
    assert "private.owner@example.com" not in serialized_headers
    assert "@" not in serialized_headers
