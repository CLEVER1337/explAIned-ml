"""The ranking service (:8002).

Same factory shape as explAIned-faiss so the two are read the same way: `create_app` takes
injectable dependencies, a lifespan wires them up, and the module-level `app` is what uvicorn
imports.

Unlike the FAISS service this one is safe to run with several workers — a LightGBM booster is
a few megabytes, not an index, so a second copy costs nothing worth optimising.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from feature_store.signature import FeatureSchemaMismatch

from ..config import Settings, get_settings
from ..features import RedisFeatureLoader
from ..logs import LOG_FORMAT
from ..metrics import ARTIFACT_AGE_SECONDS, MODEL_AGE_SECONDS, MODEL_LOADED
from ..redis_store import RecommendationStore
from .routes import router
from .state import MlflowModelLoader, ModelState

logger = logging.getLogger(__name__)

WATCHED_ARTIFACTS = (
    "article_embedding",
    "user_embedding",
    "trending",
    "features",
    "user_als_candidates",
)


def create_app(
    settings: Settings | None = None,
    store: RecommendationStore | None = None,
    loader=None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_store = store is None
        redis_store = store or RecommendationStore(settings.redis_url, settings.embedding_dim)

        app.state.settings = settings
        app.state.store = redis_store
        app.state.features = RedisFeatureLoader(redis_store)
        app.state.model = ModelState(loader or MlflowModelLoader(settings))

        result = await app.state.model.reload(force=True)

        if result.result == "signature_mismatch":
            # Fatal on purpose. Serving here would score every candidate from columns shifted
            # against their meaning, and nothing downstream would look wrong.
            raise FeatureSchemaMismatch(result.error or "model signature does not match feature_store")

        if result.result != "loaded":
            logger.warning(
                "starting without a model (%s): /rank answers 503 and the feed serves `unranked`",
                result.result,
            )

        MODEL_AGE_SECONDS.set_function(
            lambda: _age(app.state.model)
        )

        freshness = asyncio.create_task(_watch_freshness(app, settings.freshness_refresh_seconds))
        watcher = asyncio.create_task(_watch_model(app, settings.model_watch_seconds))

        try:
            yield
        finally:
            for task in (freshness, watcher):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            if owns_store:
                await redis_store.close()

    app = FastAPI(title="explAIned ranking", lifespan=lifespan)
    app.include_router(router)
    _expose_metrics(app)

    return app


def _expose_metrics(app: FastAPI) -> None:
    """`/metrics`, preferring the same instrumentation explAIned-faiss uses.

    prometheus-fastapi-instrumentator adds the `http_request_duration_seconds` family, which is
    what the shared Grafana dashboard groups by service — so it is the production path. The
    fallback exposes only the domain series from `metrics.py`: enough for the ranker's own
    alerts, and it keeps `/metrics` and the test suite working in an environment where the
    wrapper is not installed.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        from starlette.responses import Response

        logger.warning(
            "prometheus-fastapi-instrumentator is not installed; exposing domain metrics only, "
            "without the http_request_duration_seconds family"
        )

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        return

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _age(state: ModelState) -> float:
    from datetime import UTC, datetime

    bundle = state.bundle
    if bundle is None:
        MODEL_LOADED.set(0)
        return float("inf")

    return (datetime.now(UTC) - bundle.loaded_at).total_seconds()


async def _watch_model(app: FastAPI, interval_seconds: float) -> None:
    """Picks up a promotion without a restart. 0 disables it; /reload-model still works."""
    if interval_seconds <= 0:
        return

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await app.state.model.reload()
        except Exception:  # noqa: BLE001 - a watcher must never take the process down
            logger.exception("model watcher iteration failed")


async def _watch_freshness(app: FastAPI, interval_seconds: float) -> None:
    """Exports how stale each offline artifact is.

    Prometheus cannot see a scheduler's UI, and neither can the C# orchestrator. Reading the
    `rec:meta:*:updated_at` keys on a timer — rather than inside a `set_function` on scrape —
    keeps /metrics from turning into a Redis dependency.
    """
    if interval_seconds <= 0:
        return

    from datetime import UTC, datetime

    while True:
        for name in WATCHED_ARTIFACTS:
            try:
                stamp = await app.state.store.get_freshness(name)
            except Exception:  # noqa: BLE001
                continue

            age = float("inf") if stamp is None else (datetime.now(UTC) - stamp).total_seconds()
            ARTIFACT_AGE_SECONDS.labels(artifact=name).set(age)

        await asyncio.sleep(interval_seconds)


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

app = create_app()
