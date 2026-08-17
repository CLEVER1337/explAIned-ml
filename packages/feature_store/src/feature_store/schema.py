"""The feature list. This module is the single source of truth.

A `FeatureSpec` carries both the name and the function that produces the value, so a column
can never be renamed without its computation following, and the model signature written at
training time is derived from the same list the service checks at startup.

**Order is load-bearing.** `FEATURES` defines the column order of every matrix handed to
LightGBM. Append new features at the end; reordering invalidates every trained model, which
is what the signature check is there to catch.

Deliberately absent:

- **tags / hubs.** The article model has no hub or category field, and `Tags` is one delimited
  free-text string. Deciding between "introduce a hub entity" and "build features on tags" is
  an open question in todo.md section 0; until it is settled, guessing here would bake the
  wrong answer into a trained model. The seam is `ArticleContext.features` — tag features
  arrive as extra keys in the same hash, no signature surgery needed beyond appending specs.
- **ComplexityScore.** `ArticleService.SaveArticleAsync` writes a constant 0 to it. A constant
  column teaches the model nothing and costs a slot in every signature comparison.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from .context import ArticleContext, RequestContext, UserContext

# Keys inside `rec:article_features:{aid}`.
A_CLICKS_24H = "clicks_24h"
A_READS_24H = "reads_24h"
A_LIKES_24H = "likes_24h"
A_DISLIKES_24H = "dislikes_24h"
A_UNIQUE_USERS_24H = "unique_users_24h"
A_WORD_COUNT = "word_count"
A_PUBLISHED_TS = "published_ts"

# Keys inside `rec:user_features:{uid}`.
U_EVENTS_7D = "events_7d"
U_READS_7D = "reads_7d"
U_LIKES_7D = "likes_7d"
U_DISTINCT_ARTICLES_7D = "distinct_articles_7d"
U_ACTIVE_HOURS_7D = "active_hours_7d"

# Per-author affinity lives in the user hash as `author_aff:{authorId}`. A cross feature has
# nowhere else to go: it belongs to a (user, author) pair, and the user side is the smaller one.
U_AUTHOR_AFFINITY_PREFIX = "author_aff:"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    description: str
    fn: Callable[[UserContext, ArticleContext, RequestContext], float]


def _log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def _article_age_hours(_u: UserContext, a: ArticleContext, r: RequestContext) -> float:
    published = a.get(A_PUBLISHED_TS)
    if published <= 0:
        return -1.0  # unknown, and distinguishable from "published this instant"

    age = (r.now.timestamp() - published) / 3600.0
    return max(age, 0.0)


def _cosine(_u: UserContext, a: ArticleContext, _r: RequestContext) -> float:
    """Vectors are stored L2-normalised (codec.normalize), so a dot product is the cosine.

    0.0 for a cold-start user is the honest neutral value: it is what an orthogonal article
    would score, not a penalty and not a bonus.
    """
    if _u.embedding is None or a.embedding is None:
        return 0.0

    return float(_u.embedding @ a.embedding)


def _author_affinity(u: UserContext, a: ArticleContext, _r: RequestContext) -> float:
    if not a.author_id:
        return 0.0

    return _log1p(u.get(f"{U_AUTHOR_AFFINITY_PREFIX}{a.author_id}"))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        "a_clicks_24h_log",
        "Clicks on the article in the last 24h, log1p",
        lambda u, a, r: _log1p(a.get(A_CLICKS_24H)),
    ),
    FeatureSpec(
        "a_reads_24h_log",
        "Reads on the article in the last 24h, log1p",
        lambda u, a, r: _log1p(a.get(A_READS_24H)),
    ),
    FeatureSpec(
        "a_unique_users_24h_log",
        "Distinct users who touched the article in 24h — separates one obsessive reader from a crowd",
        lambda u, a, r: _log1p(a.get(A_UNIQUE_USERS_24H)),
    ),
    FeatureSpec(
        "a_read_through_rate",
        "Reads per click — how often opening the article led to reading it",
        lambda u, a, r: _ratio(a.get(A_READS_24H), a.get(A_CLICKS_24H)),
    ),
    FeatureSpec(
        "a_like_ratio",
        "Likes over likes+dislikes in 24h, 0 when nobody voted",
        lambda u, a, r: _ratio(a.get(A_LIKES_24H), a.get(A_LIKES_24H) + a.get(A_DISLIKES_24H)),
    ),
    FeatureSpec(
        "a_age_hours",
        "Hours since PublishedAt, -1 when unknown",
        _article_age_hours,
    ),
    FeatureSpec(
        "a_word_count_log",
        "Article length in words, log1p",
        lambda u, a, r: _log1p(a.get(A_WORD_COUNT)),
    ),
    FeatureSpec(
        "u_events_7d_log",
        "How active the user has been in 7 days, log1p",
        lambda u, a, r: _log1p(u.get(U_EVENTS_7D)),
    ),
    FeatureSpec(
        "u_reads_7d_log",
        "Reads by the user in 7 days, log1p",
        lambda u, a, r: _log1p(u.get(U_READS_7D)),
    ),
    FeatureSpec(
        "u_distinct_articles_7d_log",
        "Breadth of the user's reading in 7 days, log1p",
        lambda u, a, r: _log1p(u.get(U_DISTINCT_ARTICLES_7D)),
    ),
    FeatureSpec(
        "u_like_rate_7d",
        "Share of the user's events that were likes — how freely this user votes",
        lambda u, a, r: _ratio(u.get(U_LIKES_7D), u.get(U_EVENTS_7D)),
    ),
    FeatureSpec(
        "u_active_hours_7d",
        "Distinct hours in which the user was active — a session-depth proxy",
        lambda u, a, r: u.get(U_ACTIVE_HOURS_7D),
    ),
    FeatureSpec(
        "x_embedding_cosine",
        "Cosine between the user and article embeddings — the only content signal in the list",
        _cosine,
    ),
    FeatureSpec(
        "x_author_affinity_log",
        "How much this user has interacted with this article's author, log1p",
        _author_affinity,
    ),
]


def feature_names() -> list[str]:
    return [spec.name for spec in FEATURES]
