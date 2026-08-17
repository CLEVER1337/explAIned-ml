from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything addressable comes from the environment — no hardcoded hosts."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://:gavno@127.0.0.1:6379/0"

    # ClickHouse over its HTTP interface: one dependency less than a native driver,
    # and the offline jobs are not latency-bound.
    clickhouse_url: str = "http://localhost:8123"
    # Lowercase on purpose — ClickHouse identifiers are case-sensitive and the deployed
    # database is `explained`, not `explAIned`.
    clickhouse_database: str = "explained"
    clickhouse_table: str = "user_events"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""

    articles_base_url: str = "http://localhost:5036"
    identity_base_url: str = "http://localhost:5125"

    # Fixed by paraphrase-multilingual-MiniLM-L12-v2 and by EMBEDDING_DIM in explAIned-faiss.
    embedding_dim: int = 384
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # How long a derived Redis key survives an unnoticed dead job. Shorter than the
    # cadence times a few runs, so stale personalization expires instead of lingering.
    trending_ttl_seconds: int = 3600
    als_candidates_ttl_seconds: int = 172800
    features_ttl_seconds: int = 86400

    trending_window_hours: int = 24
    trending_size: int = 100

    user_embedding_window_days: int = 30
    als_candidates_per_user: int = 200

    # SQLite rather than `file:./mlruns`: the filesystem tracking *and* registry backends were
    # deprecated in Feb 2026 and warn on every call. SQLite is still zero infrastructure and is
    # what the deprecation notice points at. Artifacts still land on disk next to it.
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_registry_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_artifact_root: str = "./mlruns/artifacts"
    mlflow_experiment: str = "explained-ranking"
    mlflow_model_name: str = "explained-ranker"
    # An alias, not a file path: promoting a model must not require redeploying the service.
    mlflow_model_alias: str = "champion"

    host: str = "0.0.0.0"
    port: int = 8002

    # 0 disables the poller; the model then moves only on POST /reload-model.
    model_watch_seconds: float = 300.0
    freshness_refresh_seconds: float = 30.0

    # Clamp, not a validation rule: a bigger pool is served, not rejected. See routes.rank.
    max_candidates: int = 500

    request_timeout_seconds: float = 10.0
    index_dir: Path = Path("index_data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
