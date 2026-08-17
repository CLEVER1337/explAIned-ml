"""LightGBM itself is never called — fitting is the library's job. What is tested is the part
that decides whether the model is worth anything: where negatives come from, that degraded
impressions are excluded, and that the split is temporal rather than random.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from explained_ml.articles import Article
from explained_ml.config import Settings
from explained_ml.jobs.train_lightgbm import (
    TRUSTED_LEVELS,
    Dataset,
    _temporal_split,
    build,
)
from explained_ml.logs import EXIT_NOTHING_TO_DO, EXIT_OK
from explained_ml.redis_store import RecommendationStore
from feature_store.schema import feature_names

AS_OF = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
DIM = 4


class FakeClickHouse:
    def __init__(
        self,
        labelled=None,
        popular=None,
        impressions=None,
        has_impressions_table: bool = False,
    ) -> None:
        self.labelled = labelled or []
        self.popular = popular or []
        self.impressions = impressions or []
        self.has_impressions_table = has_impressions_table
        self.queries: list[str] = []

    async def query_rows(self, sql: str, parameters: dict | None = None):
        self.queries.append(sql)

        if "system.tables" in sql:
            return [{"n": 1 if self.has_impressions_table else 0}]
        if "feed_impressions" in sql:
            return self.impressions
        if "label" in sql:
            return self.labelled
        if "events" in sql and "ORDER BY day" in sql:
            return self.popular
        return []

    async def close(self):
        pass


class FakeArticles:
    def __init__(self, article_ids: list[str]) -> None:
        self.article_ids = article_ids

    async def iter_published(self, page_size: int = 100):
        for article_id in self.article_ids:
            yield Article(
                id=article_id,
                title="T",
                content="C",
                description="D",
                tags="ml",
                author_id="author-1",
                published_at=AS_OF,
                updated_at=AS_OF,
                word_count=100,
            )

    async def close(self):
        pass


class FakeTrainer:
    def __init__(self) -> None:
        self.train_groups: list[int] | None = None
        self.holdout_groups: list[int] | None = None
        self.feature_width: int | None = None

    def __call__(self, train, holdout, params):
        self.train_groups = list(train[2])
        self.holdout_groups = list(holdout[2])
        self.feature_width = train[0].shape[1]
        return self

    def predict(self, matrix):
        return np.zeros(len(matrix))


def labelled(user: str, article: str, day: str, label: int = 2) -> dict:
    return {"user_id": user, "article_id": article, "day": day, "label": label}


def day_str(days_ago: int) -> str:
    return (AS_OF - timedelta(days=days_ago)).date().isoformat()


@pytest.fixture
def settings():
    return Settings(embedding_dim=DIM)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", DIM, client=fake_redis)


async def run(settings, store, clickhouse, articles, trainer=None, **overrides):
    kwargs = {
        "window_days": 30,
        "holdout_days": 7,
        "negatives_mode": "sampled",
        "neg_ratio": 2,
        "min_examples": 1,
        "min_groups": 1,
        "promote": False,
        "params": {},
    }
    kwargs.update(overrides)

    return await build(
        settings,
        as_of=AS_OF,
        clickhouse=clickhouse,
        articles=articles,
        store=store,
        trainer=trainer or FakeTrainer(),
        log_to_mlflow=False,
        **kwargs,
    )


# --- grouping and splitting -----------------------------------------------------------


def test_groups_are_keyed_by_user_and_day():
    dataset = Dataset()
    dataset.add("u1", "2026-08-10", "a", 2.0)
    dataset.add("u1", "2026-08-10", "b", 0.0)
    dataset.add("u1", "2026-08-11", "c", 3.0)

    groups = dataset.groups()

    # A ranker is judged on one page at a time, so a group is one user's one day.
    assert len(groups) == 2
    assert len(groups[("u1", "2026-08-10")]) == 2


def test_the_split_is_temporal_not_random():
    days = [day_str(20), day_str(10), day_str(3), day_str(1)]

    holdout = _temporal_split(days, AS_OF, holdout_days=7)

    # A random split leaks: the same article on the same day would land on both sides.
    assert holdout == [False, False, True, True]


def test_the_holdout_boundary_is_inclusive():
    assert _temporal_split([day_str(7)], AS_OF, holdout_days=7) == [True]
    assert _temporal_split([day_str(8)], AS_OF, holdout_days=7) == [False]


# --- negatives ------------------------------------------------------------------------


async def test_sampled_negatives_come_from_popular_untouched_articles(settings, store):
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(10)), labelled("u1", "a", day_str(2))],
        popular=[
            {"day": day_str(10), "article_id": "a", "events": 50},
            {"day": day_str(10), "article_id": "b", "events": 30},
            {"day": day_str(2), "article_id": "c", "events": 20},
        ],
    )
    trainer = FakeTrainer()

    outcome = await run(settings, store, clickhouse, FakeArticles(["a", "b", "c"]), trainer)

    assert outcome == EXIT_OK
    # The engaged article must not also appear as a negative for the same user and day.
    assert sum(trainer.train_groups or []) > 1


async def test_impressions_mode_stops_when_the_table_is_absent(settings, store):
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(2))], has_impressions_table=False
    )

    outcome = await run(
        settings, store, clickhouse, FakeArticles(["a"]), negatives_mode="impressions"
    )

    # Silently falling back to sampled negatives would produce a popularity-biased model that
    # the run's parameters claim was trained on impressions.
    assert outcome == EXIT_NOTHING_TO_DO


async def test_impressions_mode_uses_shown_but_not_engaged(settings, store):
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(10)), labelled("u1", "a", day_str(2))],
        impressions=[
            {"user_id": "u1", "day": day_str(10), "article_id": "a"},
            {"user_id": "u1", "day": day_str(10), "article_id": "b"},
        ],
        has_impressions_table=True,
    )
    trainer = FakeTrainer()

    outcome = await run(
        settings,
        store,
        clickhouse,
        FakeArticles(["a", "b"]),
        trainer,
        negatives_mode="impressions",
    )

    assert outcome == EXIT_OK
    assert sum((trainer.train_groups or []) + (trainer.holdout_groups or [])) == 3


async def test_degraded_impressions_are_excluded_by_the_query(settings, store):
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(10)), labelled("u1", "a", day_str(2))],
        has_impressions_table=True,
    )

    await run(settings, store, clickhouse, FakeArticles(["a"]), negatives_mode="impressions")

    impression_sql = next(q for q in clickhouse.queries if "arrayJoin(article_ids)" in q)

    # A click on a `trending` page says nothing about the ranker — that page had no ranker.
    for level in TRUSTED_LEVELS:
        assert level in impression_sql
    assert "'trending'" not in impression_sql
    assert "'recent'" not in impression_sql


# --- guards ---------------------------------------------------------------------------


async def test_a_positives_only_set_is_refused(settings, store):
    # Found live: with --negatives impressions on a small catalog, every article the feed showed
    # had also been engaged with, so nothing became a negative. lambdarank then has nothing to
    # separate, MAP@20 comes out at exactly 1.0, and the run looks like a triumph.
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(10)), labelled("u1", "b", day_str(2))],
        impressions=[
            {"user_id": "u1", "day": day_str(10), "article_id": "a"},
            {"user_id": "u1", "day": day_str(2), "article_id": "b"},
        ],
        has_impressions_table=True,
    )

    outcome = await run(
        settings, store, clickhouse, FakeArticles(["a", "b"]), negatives_mode="impressions"
    )

    assert outcome == EXIT_NOTHING_TO_DO


async def test_a_mixed_set_trains(settings, store):
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(10)), labelled("u1", "a", day_str(2))],
        impressions=[
            {"user_id": "u1", "day": day_str(10), "article_id": "a"},
            {"user_id": "u1", "day": day_str(10), "article_id": "ignored"},
        ],
        has_impressions_table=True,
    )

    outcome = await run(
        settings, store, clickhouse, FakeArticles(["a", "ignored"]), negatives_mode="impressions"
    )

    assert outcome == EXIT_OK


async def test_no_labelled_interaction_trains_nothing(settings, store):
    outcome = await run(settings, store, FakeClickHouse(), FakeArticles(["a"]))

    assert outcome == EXIT_NOTHING_TO_DO


async def test_too_few_examples_leaves_the_champion_serving(settings, store):
    clickhouse = FakeClickHouse(labelled=[labelled("u1", "a", day_str(2))])

    outcome = await run(
        settings, store, clickhouse, FakeArticles(["a"]), min_examples=500, min_groups=50
    )

    assert outcome == EXIT_NOTHING_TO_DO


async def test_an_empty_holdout_is_refused(settings, store):
    # Everything inside the holdout window means there is nothing to train on, and vice versa.
    clickhouse = FakeClickHouse(labelled=[labelled("u1", "a", day_str(1))])

    outcome = await run(settings, store, clickhouse, FakeArticles(["a"]), holdout_days=7)

    assert outcome == EXIT_NOTHING_TO_DO


async def test_the_matrix_width_matches_the_feature_schema(settings, store):
    clickhouse = FakeClickHouse(
        labelled=[labelled("u1", "a", day_str(10)), labelled("u1", "b", day_str(2))],
        # A popular pool the user did not touch, so the sampler has negatives to draw and the
        # positives-only guard does not (correctly) refuse the run.
        popular=[
            {"day": day_str(10), "article_id": "unseen", "events": 40},
            {"day": day_str(2), "article_id": "unseen", "events": 40},
        ],
    )
    trainer = FakeTrainer()

    await run(settings, store, clickhouse, FakeArticles(["a", "b", "unseen"]), trainer)

    assert trainer.feature_width == len(feature_names())
