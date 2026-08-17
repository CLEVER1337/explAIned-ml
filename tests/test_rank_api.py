"""The service, with a fake booster and a fake registry.

LightGBM and MLflow are never touched here: `ModelBundle` only needs something with `predict`,
and the loader is an interface. What is worth testing is the contract the orchestrator relies
on — every id returned, 503 rather than a crash when there is no model, and a hard refusal when
the model's feature list disagrees with this code.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from explained_ml.config import Settings
from explained_ml.redis_store import RecommendationStore
from explained_ml.service.app import create_app
from explained_ml.service.state import ModelBundle
from explained_ml.tracking import ModelUnavailable
from feature_store.schema import feature_names
from feature_store.signature import FeatureSchemaMismatch, schema_version

DIM = 4


class FakeBooster:
    """Scores by the first feature so ordering is predictable from the inputs."""

    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, int]] = []

    def predict(self, matrix):
        self.seen_shapes.append(matrix.shape)
        return np.asarray(matrix)[:, 0]


class FakeLoader:
    def __init__(self, names=None, version: str = "1", error: Exception | None = None) -> None:
        self.names = list(names if names is not None else feature_names())
        self.version = version
        self.error = error
        self.booster = FakeBooster()
        self.loads = 0

    def current_version(self) -> str:
        if isinstance(self.error, ModelUnavailable):
            raise self.error
        return self.version

    def load(self) -> ModelBundle:
        if self.error:
            raise self.error

        from feature_store.signature import assert_matches

        assert_matches(self.names)
        self.loads += 1

        return ModelBundle(
            booster=self.booster,
            feature_names=tuple(self.names),
            name="explained-ranker",
            version=self.version,
            schema_version=schema_version(),
            loaded_at=datetime.now(UTC),
        )


@pytest.fixture
def settings():
    return Settings(embedding_dim=DIM, model_watch_seconds=0, freshness_refresh_seconds=0)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", DIM, client=fake_redis)


def client_for(settings, store, loader) -> TestClient:
    return TestClient(create_app(settings=settings, store=store, loader=loader))


def test_health_is_ok_and_describes_the_model(settings, store):
    with client_for(settings, store, FakeLoader()) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["features"] == len(feature_names())


def test_health_stays_ok_without_a_model(settings, store):
    # Liveness, not readiness: no model yet is the normal state before the first training run.
    loader = FakeLoader(error=ModelUnavailable("no alias"))

    with client_for(settings, store, loader) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_rank_returns_every_candidate(settings, store):
    with client_for(settings, store, FakeLoader()) as client:
        body = client.post("/rank", json={"user_id": "u1", "candidates": ["a", "b", "c"]}).json()

    # The orchestrator hands over its whole pool and expects it back; a short response silently
    # shortens the user's page.
    assert sorted(item["id"] for item in body["ranked"]) == ["a", "b", "c"]


async def test_rank_orders_by_descending_score(settings, store):
    # FakeBooster scores by the first feature (a_clicks_24h_log), so seeded rows decide the order.
    await store.set_article_features(
        {"a": {"clicks_24h": 1}, "b": {"clicks_24h": 100}, "c": {"clicks_24h": 10}}, 60
    )

    with client_for(settings, store, FakeLoader()) as client:
        body = client.post("/rank", json={"user_id": "u1", "candidates": ["a", "b", "c"]}).json()

    scores = [item["score"] for item in body["ranked"]]

    assert [item["id"] for item in body["ranked"]] == ["b", "c", "a"]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_candidate_list_is_an_empty_ranking(settings, store):
    with client_for(settings, store, FakeLoader()) as client:
        response = client.post("/rank", json={"user_id": "u1", "candidates": []})

    assert response.status_code == 200
    assert response.json() == {"ranked": []}


def test_rank_without_a_model_is_503(settings, store):
    loader = FakeLoader(error=ModelUnavailable("no alias"))

    with client_for(settings, store, loader) as client:
        response = client.post("/rank", json={"user_id": "u1", "candidates": ["a"]})

    # IRanker has no "disabled" outcome — a 503 is what makes the feed serve `unranked`.
    assert response.status_code == 503


def test_a_malformed_body_is_422(settings, store):
    with client_for(settings, store, FakeLoader()) as client:
        assert client.post("/rank", json={"candidates": ["a"]}).status_code == 422
        assert client.post("/rank", json={"user_id": "", "candidates": []}).status_code == 422


def test_an_oversized_pool_is_served_not_rejected(settings, store):
    small = settings.model_copy(update={"max_candidates": 2})
    ids = ["a", "b", "c", "d"]

    with client_for(small, store, FakeLoader()) as client:
        response = client.post("/rank", json={"user_id": "u1", "candidates": ids})

    # A 422 here would be recorded upstream as a ranker failure caused by the orchestrator's
    # own pool size, which is a confusing way to say "too many".
    assert response.status_code == 200
    assert sorted(item["id"] for item in response.json()["ranked"]) == ids


def test_the_model_scores_one_row_per_candidate(settings, store):
    loader = FakeLoader()

    with client_for(settings, store, loader) as client:
        client.post("/rank", json={"user_id": "u1", "candidates": ["a", "b", "c"]})

    assert loader.booster.seen_shapes[-1] == (3, len(feature_names()))


def test_a_cold_user_is_still_ranked(settings, store):
    # No embedding, no feature row: the candidates get neutral values rather than a 5xx.
    with client_for(settings, store, FakeLoader()) as client:
        response = client.post("/rank", json={"user_id": "never-seen", "candidates": ["a", "b"]})

    assert response.status_code == 200
    assert len(response.json()["ranked"]) == 2


def test_a_signature_mismatch_refuses_to_start(settings, store):
    # The one automatic guard against train/serve skew. Starting anyway would score every
    # candidate from columns shifted against their meaning, invisibly.
    loader = FakeLoader(names=[*feature_names(), "a_feature_this_code_does_not_have"])

    with pytest.raises(FeatureSchemaMismatch), client_for(settings, store, loader):
        pass


def test_reload_model_reports_its_outcome(settings, store):
    with client_for(settings, store, FakeLoader()) as client:
        body = client.post("/reload-model").json()

    assert body["result"] == "loaded"
    assert body["model_loaded"] is True


def test_reload_without_a_model_is_503(settings, store):
    loader = FakeLoader(error=ModelUnavailable("no alias"))

    with client_for(settings, store, loader) as client:
        response = client.post("/reload-model")

    assert response.status_code == 503
    assert response.json()["result"] == "missing"


def test_a_failed_reload_keeps_the_previous_model_serving(settings, store):
    loader = FakeLoader()

    with client_for(settings, store, loader) as client:
        loader.error = RuntimeError("registry unreachable")
        response = client.post("/reload-model")

        assert response.status_code == 503
        # A stale ranker beats no ranker: losing it drops every user to `unranked` at once.
        assert client.post("/rank", json={"user_id": "u1", "candidates": ["a"]}).status_code == 200


def test_metrics_exposes_the_domain_series(settings, store):
    with client_for(settings, store, FakeLoader()) as client:
        client.post("/rank", json={"user_id": "u1", "candidates": ["a"]})
        text = client.get("/metrics").text

    assert "ranking_rank_total" in text
    assert "ranking_model_loaded" in text
    assert "ranking_rank_seconds" in text
