import numpy as np
import pytest

from explained_ml.articles import Article
from explained_ml.config import Settings
from explained_ml.jobs.embeddings import build
from explained_ml.logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK
from explained_ml.redis_store import (
    ARTICLE_EMBEDDING_PREFIX,
    FINGERPRINTS_KEY,
    RecommendationStore,
    freshness_key,
)

DIM = 8


class FakeArticles:
    def __init__(self, pages: list[list[Article]], error: Exception | None = None) -> None:
        self.pages = pages
        self.error = error
        self.pages_served = 0

    async def iter_published(self, page_size: int = 100):
        if self.error:
            raise self.error

        for page in self.pages:
            self.pages_served += 1
            for article in page:
                yield article

    async def close(self):
        pass


def article(article_id: str, title: str = "Title") -> Article:
    return Article(
        id=article_id,
        title=title,
        content="Content of the article.",
        description="Summary.",
        tags="ml",
        author_id="author-1",
        published_at=None,
        updated_at=None,
        word_count=4,
    )


def full_page(prefix: str, size: int = 100) -> list[Article]:
    return [article(f"{prefix}-{i}") for i in range(size)]


@pytest.fixture
def settings():
    return Settings(embedding_dim=DIM)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", DIM, client=fake_redis)


async def test_every_article_is_encoded_and_stored(settings, store, fake_redis, encoder):
    articles = FakeArticles([[article("a1"), article("a2")]])

    assert await build(settings, encoder, articles=articles, store=store) == EXIT_OK
    assert f"{ARTICLE_EMBEDDING_PREFIX}a1" in fake_redis.strings
    assert len(fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}a1"]) == DIM * 4


async def test_stored_vectors_are_unit_length(settings, store, fake_redis, encoder):
    # x_embedding_cosine reads these directly and treats a dot product as a cosine.
    await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    raw = fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}a1"]
    vector = np.frombuffer(raw, dtype="<f4")

    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


async def test_fingerprints_are_recorded(settings, store, fake_redis, encoder):
    await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    assert fake_redis.hashes[FINGERPRINTS_KEY]


async def test_freshness_is_stamped(settings, store, fake_redis, encoder):
    await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    assert freshness_key("article_embedding") in fake_redis.strings


async def test_unchanged_articles_are_not_re_encoded(settings, store, encoder):
    articles = FakeArticles([[article("a1")]])
    await build(settings, encoder, articles=articles, store=store)
    calls_after_first = len(encoder.calls)

    outcome = await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    assert outcome == EXIT_NOTHING_TO_DO
    assert len(encoder.calls) == calls_after_first


async def test_changed_text_is_re_encoded(settings, store, encoder):
    await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    edited = FakeArticles([[article("a1", title="A different title")]])
    outcome = await build(settings, encoder, articles=edited, store=store)

    assert outcome == EXIT_OK


async def test_full_mode_re_encodes_even_unchanged_articles(settings, store, encoder):
    await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    outcome = await build(
        settings, encoder, full=True, articles=FakeArticles([[article("a1")]]), store=store
    )

    assert outcome == EXIT_OK


async def test_the_walk_stops_after_consecutive_unchanged_pages(settings, store, encoder):
    # /articles/recent is ordered newest-first and PublishedAt is stamped at creation, so once
    # a whole page is known, everything behind it is older still. Walking the rest hourly is
    # wasted work on a real catalog.
    pages = [full_page("p1"), full_page("p2"), full_page("p3"), full_page("p4")]
    await build(settings, encoder, full=True, articles=FakeArticles(pages), store=store)

    second = FakeArticles([full_page("p1"), full_page("p2"), full_page("p3"), full_page("p4")])
    await build(settings, encoder, stop_after_clean_pages=2, articles=second, store=store)

    assert second.pages_served < 4


async def test_full_mode_walks_the_whole_catalog(settings, store, encoder):
    pages = [full_page("p1"), full_page("p2"), full_page("p3")]
    await build(settings, encoder, full=True, articles=FakeArticles(pages), store=store)

    second = FakeArticles([full_page("p1"), full_page("p2"), full_page("p3")])
    await build(settings, encoder, full=True, articles=second, store=store)

    assert second.pages_served == 3


async def test_an_empty_catalog_writes_nothing(settings, store, fake_redis, encoder):
    outcome = await build(settings, encoder, articles=FakeArticles([]), store=store)

    assert outcome == EXIT_NOTHING_TO_DO
    assert not fake_redis.strings


async def test_an_empty_catalog_does_not_destroy_existing_vectors(settings, store, fake_redis, encoder):
    await build(settings, encoder, articles=FakeArticles([[article("a1")]]), store=store)

    await build(settings, encoder, articles=FakeArticles([]), store=store)

    assert f"{ARTICLE_EMBEDDING_PREFIX}a1" in fake_redis.strings


async def test_an_article_service_outage_is_infra_not_emptiness(settings, store, encoder):
    from explained_ml.articles import ArticleServiceError

    articles = FakeArticles([], error=ArticleServiceError("connection refused"))

    assert await build(settings, encoder, articles=articles, store=store) == EXIT_INFRA


async def test_a_wrong_width_vector_is_skipped_not_written(settings, store, fake_redis):
    class BadEncoder:
        dim = DIM
        calls: list = []

        def encode(self, texts):
            return np.ones((len(texts), DIM + 1), dtype=np.float32)

    outcome = await build(
        settings, BadEncoder(), articles=FakeArticles([[article("a1")]]), store=store
    )

    # A 388-float vector in a 384-dim index is a decode error in explAIned-faiss much later,
    # with no hint of where it came from.
    assert outcome == EXIT_OK
    assert f"{ARTICLE_EMBEDDING_PREFIX}a1" not in fake_redis.strings


async def test_max_articles_bounds_the_walk(settings, store, encoder):
    articles = FakeArticles([full_page("p1"), full_page("p2")])

    await build(settings, encoder, max_articles=10, articles=articles, store=store)

    assert len(encoder.calls[0]) <= 32  # batched, and far short of 200
