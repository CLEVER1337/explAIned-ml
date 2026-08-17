"""ClickHouse queries, in one place so they can be read as SQL rather than as string literals.

Two rules hold across every query here:

**`FINAL`.** `explained.user_events` is a `ReplacingMergeTree` keyed on
`(user_id, occurred_at, event_id)`, and it only collapses duplicates when parts merge. The
consumer commits Kafka offsets after the ClickHouse insert, so a crash between the two replays
rows. Without `FINAL` every aggregate silently over-counts by however much got retried. If this
ever becomes slow, the fix is a materialized view (todo.md section 1), not dropping `FINAL`.

**`as_of` is always a parameter.** The same SQL serves the materialization job (`as_of = now`)
and `train_lightgbm.py` (`as_of` = the moment of the impression being learned from). A query
that hardcoded `now()` would make training rows describe the present rather than the past,
which produces a model that looks excellent offline and does nothing online.

Event weights are shared deliberately: trending, user embeddings and ALS should disagree about
windows and decay, not about what a like is worth relative to a click.
"""

EVENT_WEIGHT_SQL = """multiIf(
        event_type = 'ArticleShared',    6.0,
        event_type = 'ArticleCommented', 5.0,
        event_type = 'ArticleLiked',     4.0,
        event_type = 'ArticleRead',      3.0,
        event_type = 'ArticleClicked',   1.0,
        event_type = 'ArticleDisliked', -4.0,
        0.0)"""


def trending(table: str) -> str:
    """Time-decayed popularity. `min_reach` is the anti-gaming term: one enthusiastic user
    must not be able to pin an article to the top of everybody's feed."""
    return f"""
SELECT
    article_id,
    sum({EVENT_WEIGHT_SQL} * pow(0.5, dateDiff('hour', occurred_at, {{as_of:DateTime64(3)}})
                                       / {{half_life_hours:Float64}})) AS score,
    uniqExact(user_id) AS reach
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_hours:UInt32}} HOUR
  AND article_id != ''
GROUP BY article_id
HAVING reach >= {{min_reach:UInt32}} AND score > 0
ORDER BY score DESC, reach DESC, article_id ASC
LIMIT {{limit:UInt32}}
"""


def user_affinity(table: str) -> str:
    """Per-(user, article) affinity, newest first, capped per user.

    Negative totals are dropped rather than subtracted: a disliked article should not pull the
    user's centroid toward its opposite, which is a direction in embedding space that means
    nothing. Pushing away is a later iteration.
    """
    return f"""
SELECT
    user_id,
    article_id,
    sum({EVENT_WEIGHT_SQL}) AS affinity,
    max(occurred_at) AS last_at
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND user_id != '' AND article_id != ''
GROUP BY user_id, article_id
HAVING affinity > 0
ORDER BY user_id ASC, last_at DESC
LIMIT {{per_user:UInt32}} BY user_id
"""


def article_stats(table: str) -> str:
    return f"""
SELECT
    article_id,
    countIf(event_type = 'ArticleClicked')   AS clicks,
    countIf(event_type = 'ArticleRead')      AS reads,
    countIf(event_type = 'ArticleLiked')     AS likes,
    countIf(event_type = 'ArticleDisliked')  AS dislikes,
    uniqExact(user_id)                       AS unique_users
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_hours:UInt32}} HOUR
  AND article_id != ''
GROUP BY article_id
"""


def user_stats(table: str) -> str:
    return f"""
SELECT
    user_id,
    count()                                 AS events,
    countIf(event_type = 'ArticleRead')     AS reads,
    countIf(event_type = 'ArticleLiked')    AS likes,
    uniqExact(article_id)                   AS distinct_articles,
    uniqExact(toStartOfHour(occurred_at))   AS active_hours
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND user_id != ''
GROUP BY user_id
"""


def user_author_affinity(table: str) -> str:
    """Positive interactions per (user, article), for folding into per-author counts in Python.

    The join to `authorId` happens over HTTP rather than in SQL because ClickHouse holds no
    article metadata — the article service owns it, and this repository does not read another
    service's PostgreSQL.
    """
    return f"""
SELECT
    user_id,
    article_id,
    count() AS positives
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND user_id != '' AND article_id != ''
  AND event_type IN ('ArticleRead', 'ArticleLiked', 'ArticleShared', 'ArticleCommented')
GROUP BY user_id, article_id
"""


def interactions(table: str) -> str:
    """The ALS training matrix: every positive (user, article) pair in the window."""
    return f"""
SELECT
    user_id,
    article_id,
    sum({EVENT_WEIGHT_SQL}) AS affinity
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND user_id != '' AND article_id != ''
GROUP BY user_id, article_id
HAVING affinity > 0
"""


def labelled_events(table: str) -> str:
    """Positive labels for ranking: what each user engaged with, and how strongly.

    Graded rather than binary — `lambdarank` can use the gradation, and "read then liked" is a
    genuinely stronger signal than "clicked and bounced".
    """
    return f"""
SELECT
    user_id,
    article_id,
    toDate(occurred_at) AS day,
    max(multiIf(
        event_type IN ('ArticleLiked', 'ArticleShared', 'ArticleCommented'), 3,
        event_type = 'ArticleRead',                                          2,
        event_type = 'ArticleClicked',                                       1,
        0)) AS label,
    min(occurred_at) AS first_at
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND user_id != '' AND article_id != ''
GROUP BY user_id, article_id, day
HAVING label > 0
"""


def popular_by_day(table: str) -> str:
    """Popularity per day, used to draw negatives when `feed_impressions` does not exist yet.

    This is a *biased* negative sampler — it teaches the model that popular-and-not-clicked is
    negative, which correlates with popularity itself. Real negatives come from impressions;
    see `train_lightgbm.py --negatives impressions`.
    """
    return f"""
SELECT
    toDate(occurred_at) AS day,
    article_id,
    count() AS events
FROM {table} FINAL
WHERE occurred_at <  {{as_of:DateTime64(3)}}
  AND occurred_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND article_id != ''
GROUP BY day, article_id
ORDER BY day ASC, events DESC
LIMIT {{per_day:UInt32}} BY day
"""
