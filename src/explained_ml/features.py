"""Building feature rows, and loading them back.

`feature_store` decides what a feature *is*; this module decides where the numbers come from.
There are two loaders and they must produce identical rows:

    offline (ClickHouse + article service, `as_of`-aware)  ->  training, and the materializer
    online  (Redis, one round trip)                        ->  /rank

The loop is closed on purpose: `build_features` runs the *offline* loader at `as_of = now` and
writes its output to Redis, so an online row is by construction an offline row that took a
detour through Redis. If the two ever disagree it is a bug in serialization, not in arithmetic
— and `encode_row`/`decode_row` are the single serializer for both directions.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from feature_store.context import ArticleContext, UserContext
from feature_store.schema import (
    A_CLICKS_24H,
    A_DISLIKES_24H,
    A_LIKES_24H,
    A_PUBLISHED_TS,
    A_READS_24H,
    A_UNIQUE_USERS_24H,
    A_WORD_COUNT,
    U_ACTIVE_HOURS_7D,
    U_AUTHOR_AFFINITY_PREFIX,
    U_DISTINCT_ARTICLES_7D,
    U_EVENTS_7D,
    U_LIKES_7D,
    U_READS_7D,
)

from .articles import Article, ArticleClient
from .clickhouse import ClickHouseClient
from .redis_store import RecommendationStore
from .sql import article_stats, user_author_affinity, user_stats

logger = logging.getLogger(__name__)

# Only the strongest author affinities are kept: the row is read on every /rank, and the tail
# of a long-tail distribution costs bytes without changing a score.
MAX_AUTHORS_PER_USER = 32


@dataclass(frozen=True, slots=True)
class FeatureRows:
    articles: dict[str, dict[str, float]]
    users: dict[str, dict[str, float]]
    authors: dict[str, str]


class OfflineFeatureBuilder:
    """ClickHouse + the article service -> plain feature dicts, as of an arbitrary moment."""

    def __init__(
        self,
        clickhouse: ClickHouseClient,
        articles: ArticleClient,
        table: str,
    ) -> None:
        self._clickhouse = clickhouse
        self._articles = articles
        self._table = table

    async def build(
        self,
        as_of: datetime,
        article_window_hours: int = 24,
        user_window_days: int = 7,
    ) -> FeatureRows:
        catalog = [a async for a in self._articles.iter_published()]
        authors = {a.id: a.author_id for a in catalog}

        article_rows = self._article_rows(catalog, as_of)
        for row in await self._clickhouse.query_rows(
            article_stats(self._table), {"as_of": as_of, "window_hours": article_window_hours}
        ):
            article_id = str(row.get("article_id") or "")
            if article_id in article_rows:
                article_rows[article_id].update(
                    {
                        A_CLICKS_24H: float(row.get("clicks", 0)),
                        A_READS_24H: float(row.get("reads", 0)),
                        A_LIKES_24H: float(row.get("likes", 0)),
                        A_DISLIKES_24H: float(row.get("dislikes", 0)),
                        A_UNIQUE_USERS_24H: float(row.get("unique_users", 0)),
                    }
                )

        user_rows: dict[str, dict[str, float]] = {}
        for row in await self._clickhouse.query_rows(
            user_stats(self._table), {"as_of": as_of, "window_days": user_window_days}
        ):
            user_id = str(row.get("user_id") or "")
            if not user_id:
                continue

            user_rows[user_id] = {
                U_EVENTS_7D: float(row.get("events", 0)),
                U_READS_7D: float(row.get("reads", 0)),
                U_LIKES_7D: float(row.get("likes", 0)),
                U_DISTINCT_ARTICLES_7D: float(row.get("distinct_articles", 0)),
                U_ACTIVE_HOURS_7D: float(row.get("active_hours", 0)),
            }

        await self._fold_author_affinity(user_rows, authors, as_of, user_window_days)

        return FeatureRows(articles=article_rows, users=user_rows, authors=authors)

    def _article_rows(self, catalog: Sequence[Article], as_of: datetime) -> dict[str, dict[str, float]]:
        return {
            article.id: {
                A_WORD_COUNT: float(article.word_count),
                A_PUBLISHED_TS: article.published_at.timestamp() if article.published_at else 0.0,
                A_CLICKS_24H: 0.0,
                A_READS_24H: 0.0,
                A_LIKES_24H: 0.0,
                A_DISLIKES_24H: 0.0,
                A_UNIQUE_USERS_24H: 0.0,
            }
            for article in catalog
        }

    async def _fold_author_affinity(
        self,
        user_rows: dict[str, dict[str, float]],
        authors: dict[str, str],
        as_of: datetime,
        window_days: int,
    ) -> None:
        """(user, article) positives -> (user, author) counts.

        The join happens here rather than in SQL because ClickHouse holds no article metadata;
        the article service owns it and this repository does not read another service's
        PostgreSQL.
        """
        rows = await self._clickhouse.query_rows(
            user_author_affinity(self._table), {"as_of": as_of, "window_days": window_days}
        )

        per_user: dict[str, dict[str, float]] = {}
        for row in rows:
            user_id = str(row.get("user_id") or "")
            author_id = authors.get(str(row.get("article_id") or ""))
            if not user_id or not author_id:
                continue

            bucket = per_user.setdefault(user_id, {})
            bucket[author_id] = bucket.get(author_id, 0.0) + float(row.get("positives", 0))

        for user_id, affinities in per_user.items():
            top = sorted(affinities.items(), key=lambda kv: kv[1], reverse=True)[:MAX_AUTHORS_PER_USER]
            row = user_rows.setdefault(user_id, {})
            for author_id, count in top:
                row[f"{U_AUTHOR_AFFINITY_PREFIX}{author_id}"] = count


class RedisFeatureLoader:
    """The online path. Everything /rank needs in as few round trips as Redis allows."""

    def __init__(self, store: RecommendationStore) -> None:
        self._store = store

    async def load(
        self, user_id: str, article_ids: list[str], authors: dict[str, str] | None = None
    ) -> tuple[UserContext, list[ArticleContext]]:
        user_features = await self._store.get_user_features(user_id)
        user_embedding = await self._store.get_user_embedding(user_id)

        article_features = await self._store.get_article_features(article_ids)
        article_embeddings = await self._store.get_article_embeddings(article_ids)

        user = UserContext(user_id=user_id, features=user_features, embedding=user_embedding)

        # A candidate whose row expired is scored on neutral values rather than dropped: the
        # orchestrator asked for these ids and expects all of them back.
        articles = [
            ArticleContext(
                article_id=article_id,
                author_id=(authors or {}).get(article_id, ""),
                features=article_features.get(article_id, {}),
                embedding=article_embeddings.get(article_id),
            )
            for article_id in article_ids
        ]

        missing = sum(1 for a in articles if not a.features)
        if missing:
            logger.debug("%d of %d candidates had no feature row", missing, len(articles))

        return user, articles


def contexts_from_rows(
    user_id: str,
    article_ids: list[str],
    rows: FeatureRows,
    user_embedding=None,
    article_embeddings: dict | None = None,
) -> tuple[UserContext, list[ArticleContext]]:
    """The offline mirror of `RedisFeatureLoader.load`, for training."""
    user = UserContext(
        user_id=user_id, features=rows.users.get(user_id, {}), embedding=user_embedding
    )

    articles = [
        ArticleContext(
            article_id=article_id,
            author_id=rows.authors.get(article_id, ""),
            features=rows.articles.get(article_id, {}),
            embedding=(article_embeddings or {}).get(article_id),
        )
        for article_id in article_ids
    ]

    return user, articles
