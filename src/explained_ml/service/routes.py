"""`POST /rank` and friends.

The response contract is narrow and worth stating plainly: **every id that comes in comes back
out**, in descending score order. The orchestrator built that candidate pool from its own
sources and expects to hand the whole thing to the ranker; a service that silently drops
candidates shortens the user's page and nothing upstream can tell why.

There is no 204 here. `FaissCandidateSource` has a "disabled" outcome for a cold user, but
`IRanker` has only success or failure — a failed rank means the feed serves `unranked`, which
is the correct answer when there is no model.
"""

import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from feature_store.compute import build_matrix
from feature_store.context import RequestContext

from ..metrics import RANK_CANDIDATES, RANK_SECONDS, RANK_TOTAL

logger = logging.getLogger(__name__)

router = APIRouter()


class RankRequest(BaseModel):
    user_id: str = Field(min_length=1)
    # No max_length: a pool larger than expected must degrade, never 422. The orchestrator
    # would record a malformed-request failure and blame the ranker for its own pool size.
    candidates: list[str] = Field(default_factory=list)


class RankedItem(BaseModel):
    id: str
    score: float


class RankResponse(BaseModel):
    ranked: list[RankedItem]


@router.post("/rank")
async def rank(request: Request, payload: RankRequest):
    state = request.app.state.model
    loader = request.app.state.features
    settings = request.app.state.settings

    if not payload.candidates:
        RANK_TOTAL.labels(result="empty").inc()
        return RankResponse(ranked=[])

    bundle = state.bundle
    if bundle is None:
        RANK_TOTAL.labels(result="no_model").inc()
        return JSONResponse(
            {"detail": "no model is loaded"}, status_code=503
        )

    candidates = payload.candidates[: settings.max_candidates]
    overflow = payload.candidates[settings.max_candidates :]
    RANK_CANDIDATES.observe(len(payload.candidates))

    started = time.perf_counter()

    try:
        user, articles = await loader.load(payload.user_id, candidates)
    except RedisError as exc:
        RANK_TOTAL.labels(result="store_error").inc()
        logger.warning("feature load failed: %s", exc)
        return JSONResponse({"detail": "feature store unavailable"}, status_code=503)

    matrix = build_matrix(user, articles, RequestContext(now=datetime.now(UTC)))
    scores = bundle.predict(matrix)

    ordered = sorted(
        zip(candidates, (float(s) for s in scores), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    # Anything past the cap is appended at the bottom rather than dropped: the contract is
    # "every id comes back", and a truncated response is indistinguishable upstream from a
    # ranker that simply disliked those articles.
    ranked = [RankedItem(id=article_id, score=score) for article_id, score in ordered]
    ranked.extend(RankedItem(id=article_id, score=0.0) for article_id in overflow)

    RANK_SECONDS.observe(time.perf_counter() - started)
    RANK_TOTAL.labels(result="ok").inc()

    return RankResponse(ranked=ranked)


@router.post("/reload-model")
async def reload_model(request: Request):
    result = await request.app.state.model.reload(force=True)
    body = {"result": result.result, "error": result.error, **request.app.state.model.describe()}

    if result.result in ("loaded", "unchanged"):
        return body

    return JSONResponse(body, status_code=503)


@router.get("/health")
async def health(request: Request):
    # Liveness, not readiness: a service with no model yet is alive and correctly answering
    # 503 on /rank. Failing health here would make a normal pre-first-training state look
    # like a crash loop.
    return {"status": "ok", **request.app.state.model.describe()}
