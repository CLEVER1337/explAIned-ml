import numpy as np
import pytest

from feature_store import schema
from feature_store.context import ArticleContext, RequestContext, UserContext
from feature_store.signature import FeatureSchemaMismatch, assert_matches, schema_version


def test_names_are_unique_and_ordered():
    names = schema.feature_names()

    assert len(names) == len(set(names)), "a duplicate name would silently shadow a column"
    assert names == [spec.name for spec in schema.FEATURES]


def test_every_feature_has_a_description():
    # The description is what a future reader has instead of the conversation that produced it.
    assert all(spec.description.strip() for spec in schema.FEATURES)


def test_complexity_score_is_not_a_feature():
    # The article service writes a constant 0 to it; a constant column teaches nothing.
    assert not any("complexity" in name for name in schema.feature_names())


def test_viewed_is_not_a_feature():
    # It would leak: every training positive is viewed by construction, and the orchestrator
    # already subtracts rec:user_viewed before the ranker is ever called.
    assert not any("viewed" in name for name in schema.feature_names())


def test_schema_version_is_stable_and_short():
    assert schema_version() == schema_version()
    assert len(schema_version()) == 12


def test_schema_version_changes_when_the_order_changes(monkeypatch):
    before = schema_version()
    monkeypatch.setattr(schema, "FEATURES", list(reversed(schema.FEATURES)))

    assert schema_version() != before


def test_assert_matches_accepts_the_current_list():
    assert_matches(schema.feature_names())


def test_assert_matches_names_the_first_difference():
    wrong = schema.feature_names()
    wrong[2] = "renamed_somewhere_else"

    with pytest.raises(FeatureSchemaMismatch, match="index 2"):
        assert_matches(wrong)


def test_assert_matches_reports_a_length_difference():
    with pytest.raises(FeatureSchemaMismatch, match="expects 2 features"):
        assert_matches(schema.feature_names()[:2])


def test_reordering_is_caught_even_though_the_set_is_equal():
    swapped = schema.feature_names()
    swapped[0], swapped[1] = swapped[1], swapped[0]

    with pytest.raises(FeatureSchemaMismatch):
        assert_matches(swapped)


def test_author_affinity_reads_the_per_author_key(now):
    user = UserContext("u1", features={f"{schema.U_AUTHOR_AFFINITY_PREFIX}author-7": 6.0})
    article = ArticleContext("a1", author_id="author-7")
    other = ArticleContext("a2", author_id="author-9")

    from feature_store.compute import explain

    assert explain(user, article, now)["x_author_affinity_log"] > 0
    assert explain(user, other, now)["x_author_affinity_log"] == 0


def test_cosine_uses_the_embeddings(now):
    vector = np.array([1.0, 0.0], dtype=np.float32)
    user = UserContext("u1", embedding=vector)
    aligned = ArticleContext("a1", embedding=vector)
    orthogonal = ArticleContext("a2", embedding=np.array([0.0, 1.0], dtype=np.float32))

    from feature_store.compute import explain

    assert explain(user, aligned, now)["x_embedding_cosine"] == pytest.approx(1.0)
    assert explain(user, orthogonal, now)["x_embedding_cosine"] == pytest.approx(0.0)


def test_cold_user_scores_neutral_rather_than_negative(now):
    from feature_store.compute import explain

    cold = explain(UserContext("new"), ArticleContext("a1", embedding=np.ones(2, np.float32)), now)

    assert cold["x_embedding_cosine"] == 0.0


@pytest.fixture
def now():
    from datetime import UTC, datetime

    return RequestContext(now=datetime(2026, 8, 17, 12, 0, tzinfo=UTC))
