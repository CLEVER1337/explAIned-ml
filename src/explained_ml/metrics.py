"""Prometheus series owned by the ranking service.

HTTP-level series (`http_request_duration_seconds` and friends) come from
prometheus-fastapi-instrumentator; everything here is about the model itself.

Offline jobs deliberately export nothing: they are short-lived processes, and a push
gateway would be a whole extra dependency. Their freshness is observable through the
`rec:*:updated_at` keys instead (see redis_store.freshness_key).
"""

from prometheus_client import Counter, Gauge, Histogram

MODEL_LOADED = Gauge("ranking_model_loaded", "1 when a model is serving, 0 otherwise")

MODEL_FEATURES = Gauge("ranking_model_features", "Number of features the loaded model expects")

MODEL_AGE_SECONDS = Gauge(
    "ranking_model_age_seconds",
    "Seconds since the loaded model was trained — pairs with faiss_index_age_seconds",
)

RANK_SECONDS = Histogram(
    "ranking_rank_seconds",
    "Time spent inside /rank, feature loading included",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.12, 0.25),
)

RANK_TOTAL = Counter(
    "ranking_rank_total",
    "Rank requests by outcome",
    labelnames=("result",),  # ok | empty | no_model | store_error
)

RANK_CANDIDATES = Histogram(
    "ranking_candidates",
    "Candidates per /rank request — the orchestrator's pool size as we actually see it",
    buckets=(1, 10, 25, 50, 100, 200, 400),
)

MODEL_RELOAD_TOTAL = Counter(
    "ranking_model_reload_total",
    "Model reloads by outcome",
    labelnames=("result",),  # loaded | unchanged | missing | failed | signature_mismatch
)

ARTIFACT_AGE_SECONDS = Gauge(
    "ml_artifact_age_seconds",
    "Seconds since each offline artifact was refreshed, read from rec:meta:*:updated_at",
    labelnames=("artifact",),  # article_embedding | user_embedding | trending | features | ...
)
