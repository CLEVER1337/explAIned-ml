from datetime import UTC, datetime

import numpy as np
import pytest

from feature_store import schema
from feature_store.compute import build_matrix, build_row, explain
from feature_store.context import ArticleContext, RequestContext, UserContext

NOW = RequestContext(now=datetime(2026, 8, 17, 12, 0, tzinfo=UTC))


def article(article_id: str = "a1", **features) -> ArticleContext:
    return ArticleContext(article_id, author_id="author-1", features=features)


def test_matrix_shape_matches_the_schema():
    matrix = build_matrix(UserContext("u1"), [article("a1"), article("a2")], NOW)

    assert matrix.shape == (2, len(schema.FEATURES))
    assert matrix.dtype == np.float32


def test_matrix_rows_are_the_single_row_function():
    user = UserContext("u1", features={schema.U_EVENTS_7D: 12})
    articles = [article("a1", clicks_24h=5), article("a2", clicks_24h=50)]

    matrix = build_matrix(user, articles, NOW)

    for index, item in enumerate(articles):
        assert np.array_equal(matrix[index], build_row(user, item, NOW))


def test_matrix_row_order_follows_the_candidate_order():
    # The service zips scores back onto ids by position; a reordering here would mis-attribute
    # every score in the response.
    user = UserContext("u1")
    matrix = build_matrix(user, [article("a1", clicks_24h=1), article("a2", clicks_24h=100)], NOW)

    clicks_column = schema.feature_names().index("a_clicks_24h_log")

    assert matrix[0][clicks_column] < matrix[1][clicks_column]


def test_empty_candidate_list_yields_an_empty_matrix_of_the_right_width():
    matrix = build_matrix(UserContext("u1"), [], NOW)

    assert matrix.shape == (0, len(schema.FEATURES))


def test_missing_features_produce_finite_neutral_values():
    # A candidate whose feature row expired must still be scoreable — the ranker never 500s
    # because a hash fell out of Redis.
    matrix = build_matrix(UserContext("cold"), [ArticleContext("a1")], NOW)

    assert np.isfinite(matrix).all()


def test_non_finite_feature_values_are_rejected_loudly(monkeypatch):
    # LightGBM silently treats NaN as "missing", which hides the bug that produced it.
    broken = schema.FeatureSpec("broken", "always NaN", lambda u, a, r: float("nan"))
    monkeypatch.setattr(schema, "FEATURES", [*schema.FEATURES, broken])

    import feature_store.compute as compute_module

    monkeypatch.setattr(compute_module, "FEATURES", schema.FEATURES)

    with pytest.raises(ValueError, match="broken"):
        build_matrix(UserContext("u1"), [article()], NOW)


def test_article_age_is_measured_from_the_request_clock():
    published = datetime(2026, 8, 17, 6, 0, tzinfo=UTC).timestamp()
    values = explain(UserContext("u1"), article(published_ts=published), NOW)

    assert values["a_age_hours"] == pytest.approx(6.0)


def test_unknown_publication_date_is_distinguishable_from_brand_new():
    values = explain(UserContext("u1"), article(), NOW)

    assert values["a_age_hours"] == -1.0


def test_age_never_goes_negative_for_a_future_timestamp():
    # Clock skew between the article service and this job must not invent a new value range.
    future = datetime(2026, 8, 18, tzinfo=UTC).timestamp()

    assert explain(UserContext("u1"), article(published_ts=future), NOW)["a_age_hours"] == 0.0


def test_ratios_are_zero_when_the_denominator_is_zero():
    values = explain(UserContext("u1"), article(reads_24h=0, clicks_24h=0), NOW)

    assert values["a_read_through_rate"] == 0.0
    assert values["a_like_ratio"] == 0.0


def test_like_ratio_weighs_likes_against_dislikes():
    values = explain(UserContext("u1"), article(likes_24h=3, dislikes_24h=1), NOW)

    assert values["a_like_ratio"] == pytest.approx(0.75)


def test_explain_covers_every_feature_by_name():
    values = explain(UserContext("u1"), article(), NOW)

    assert sorted(values) == sorted(schema.feature_names())


def test_computation_is_deterministic():
    user = UserContext("u1", features={schema.U_EVENTS_7D: 3})
    first = build_matrix(user, [article(clicks_24h=2)], NOW)
    second = build_matrix(user, [article(clicks_24h=2)], NOW)

    assert np.array_equal(first, second)


def test_the_clock_is_injected_not_read():
    # Training reconstructs rows as of a past moment; if a feature read datetime.now() the
    # model would learn values that were never true when the impression happened.
    earlier = RequestContext(now=datetime(2026, 8, 17, 8, 0, tzinfo=UTC))
    published = datetime(2026, 8, 17, 6, 0, tzinfo=UTC).timestamp()

    assert explain(UserContext("u"), article(published_ts=published), earlier)["a_age_hours"] == 2.0
