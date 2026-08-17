import httpx
import pytest

from explained_ml.articles import ArticleClient, ArticleServiceError


def article_json(article_id: str, **overrides) -> dict:
    payload = {
        "id": article_id,
        "title": f"Title {article_id}",
        "content": "Body paragraph.",
        "description": "Summary.",
        "tags": "kafka, distributed systems",
        "authorId": "author-1",
        "publishedAt": "2026-08-17T09:00:00Z",
        "updatedAt": "2026-08-17T09:00:00Z",
        "wordCount": 2,
    }
    payload.update(overrides)
    return payload


def client_over(handler) -> ArticleClient:
    transport = httpx.MockTransport(handler)
    return ArticleClient(
        "http://articles",
        client=httpx.AsyncClient(transport=transport, base_url="http://articles"),
    )


def paged(pages: list[list[dict]]):
    """Serve `/articles/recent` from a list of pages, keyed by offset."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params.get("offset", 0))
        limit = int(request.url.params.get("limit", 100))
        index = offset // limit if limit else 0
        body = pages[index] if index < len(pages) else []
        return httpx.Response(200, json=body)

    return handler, requests


async def test_recent_parses_the_camel_case_payload():
    handler, _ = paged([[article_json("a1")]])

    articles = await client_over(handler).recent(limit=100)

    assert articles[0].id == "a1"
    assert articles[0].author_id == "author-1"
    assert articles[0].word_count == 2
    assert articles[0].published_at is not None


async def test_iter_published_walks_every_page():
    full = [article_json(f"a{i}") for i in range(3)]
    handler, requests = paged([full, full, [article_json("last")]])

    seen = [a.id async for a in client_over(handler).iter_published(page_size=3)]

    assert seen[-1] == "last"
    assert len(seen) == 7
    assert len(requests) == 3


async def test_iter_published_stops_on_a_short_page():
    # A short page means the end of the catalog; asking for the next offset would be a wasted
    # round trip on every run of an hourly job.
    handler, requests = paged([[article_json("a1")]])

    _ = [a async for a in client_over(handler).iter_published(page_size=100)]

    assert len(requests) == 1


async def test_iter_published_stops_on_an_empty_catalog():
    handler, requests = paged([[]])

    assert [a async for a in client_over(handler).iter_published()] == []
    assert len(requests) == 1


async def test_batch_chunks_at_the_server_limit():
    seen_chunks: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = request.url.params["ids"].split(",")
        seen_chunks.append(len(ids))
        return httpx.Response(200, json=[article_json(i) for i in ids])

    ids = [f"a{i}" for i in range(250)]
    articles = await client_over(handler).get_batch(ids)

    # /articles/batch rejects more than 100 ids with a 400.
    assert seen_chunks == [100, 100, 50]
    assert [a.id for a in articles] == ids


async def test_batch_of_nothing_makes_no_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request expected")

    assert await client_over(handler).get_batch([]) == []


async def test_missing_ids_are_simply_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[article_json("a1")])

    articles = await client_over(handler).get_batch(["a1", "archived"])

    assert [a.id for a in articles] == ["a1"]


async def test_a_non_200_is_an_infrastructure_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(ArticleServiceError):
        await client_over(handler).recent()


async def test_a_transport_failure_is_an_infrastructure_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ArticleServiceError):
        await client_over(handler).recent()


async def test_embedding_text_puts_the_title_first():
    handler, _ = paged([[article_json("a1")]])
    article = (await client_over(handler).recent())[0]

    # MiniLM truncates at 128 tokens; the title carries the most topical signal per token.
    assert article.embedding_text.startswith("Title a1")
    assert "Body paragraph." in article.embedding_text


async def test_embedding_text_skips_empty_parts():
    handler, _ = paged([[article_json("a1", description="", content="")]])
    article = (await client_over(handler).recent())[0]

    assert article.embedding_text == "Title a1"


async def test_tags_are_split_from_the_single_delimited_string():
    handler, _ = paged([[article_json("a1", tags="Kafka; Distributed Systems|ml")]])
    article = (await client_over(handler).recent())[0]

    assert article.tag_list == ["kafka", "distributed systems", "ml"]


async def test_an_unparseable_timestamp_does_not_kill_the_job():
    handler, _ = paged([[article_json("a1", publishedAt="not a date")]])
    article = (await client_over(handler).recent())[0]

    assert article.published_at is None
