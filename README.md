# explAIned-ml

The Python half of the recommendation loop that is not FAISS: the ranking service on `:8002`,
the `feature_store` package, and the offline jobs that fill Redis.

The split from `explAIned-faiss` runs along the `faiss` library. Anything importing it — the
search service and the index builder — lives there. Everything else lives here, and the two
repositories talk only through Redis and HTTP.

## What it does

```
        explAIned-ml                                  Redis                     readers
┌──────────────────────────┐
│ embeddings.py       (1h) │──▶ rec:article_embedding:{aid} ──────▶ explAIned-faiss index builder
│ build_user_embeddings.py │──▶ rec:user_embedding:{uid}    ──────▶ FAISS :8001  /search
│                     (1h) │
│ update_trending.py (10m) │──▶ rec:trending:top100         ──────▶ orchestrator :5056
│ train_als.py        (1d) │──▶ rec:user_als_candidates:*   ──────▶ AlsCandidateSource
│ build_features.py   (1h) │──▶ rec:{article,user}_features ──┐
│ train_lightgbm.py   (1w) │──▶ MLflow registry ────────────┐ │
└──────────────────────────┘                                │ │
                                                            ▼ ▼
                                              ranking service :8002  POST /rank
                                                            ▲
                                                            │ ids only
                                                  orchestrator :5056
```

Inputs: `explained.user_events` in ClickHouse (behaviour) and `GET /articles/recent` on the
article service (content). Nothing here reads another service's PostgreSQL.

## Why the feed cares

The orchestrator's degradation ladder is `personalized → partial → unranked → trending →
recent → cached → empty`, and before this repository existed the top four rungs were all
unreachable: no key was ever written for them.

| Rung | What it needs | Job |
|---|---|---|
| `personalized` / `partial` | a ranker | `train_lightgbm.py` + the `:8002` service |
| `unranked` | candidates | `embeddings.py` + `build_user_embeddings.py` (FAISS), `train_als.py` |
| `trending` | `rec:trending:top100` | `update_trending.py` |

## Running

Python 3.12 or 3.13 — torch publishes no cp314 wheels, the same ceiling `explAIned-faiss` has.

```bash
poetry env use ~/.local/share/uv/python/cpython-3.13.5-*/bin/python3.13
poetry install                 # ranking service + tests
poetry install --with jobs     # adds torch, sentence-transformers, implicit (~2.5 GB)
```

First run, in order — each step is verifiable before the next one matters:

```bash
# 0. is anything even there?
poetry run check-env

# 1. a corpus worth embedding. The live one has 4-character bodies and one author.
poetry run seed-dev-corpus --articles 200 --users 12
poetry run simulate-behavior --events-per-user 250 --days 21 --direct-clickhouse

# 2. trending — the rung that serves everyone, including users with no history
poetry run update-trending --min-reach 2

# 3. content: embeddings here, index in the sibling repo
poetry run embeddings --full
poetry run build-user-embeddings
(cd ../explAIned-faiss && poetry run python -m explained_faiss.jobs.build_index)

# 4. features, then models
poetry run build-features
poetry run train-als
poetry run train-lightgbm --promote

# 5. serve
poetry run uvicorn explained_ml.service.app:app --port 8002
```

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://:gavno@127.0.0.1:6379/0` | same instance as everything else |
| `CLICKHOUSE_URL` | `http://localhost:8123` | HTTP interface |
| `CLICKHOUSE_DATABASE` | `explained` | lowercase; identifiers are case-sensitive |
| `CLICKHOUSE_TABLE` | `user_events` | behavioural events |
| `ARTICLES_BASE_URL` | `http://localhost:5036` | article service |
| `EMBEDDING_DIM` | `384` | must equal the FAISS service's setting |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fixes the dimension |
| `TRENDING_TTL_SECONDS` | `3600` | stale trending expires rather than lingering |
| `FEATURES_TTL_SECONDS` | `86400` | a dead materializer degrades to neutral features |
| `ALS_CANDIDATES_TTL_SECONDS` | `172800` | survives a couple of missed daily runs |
| `MLFLOW_TRACKING_URI` / `MLFLOW_REGISTRY_URI` | `sqlite:///mlruns/mlflow.db` | the file backends are deprecated |
| `MLFLOW_MODEL_NAME` / `MLFLOW_MODEL_ALIAS` | `explained-ranker` / `champion` | resolved at startup |
| `MODEL_WATCH_SECONDS` | `300` | picks up a promotion without a restart; `0` disables |
| `MAX_CANDIDATES` | `500` | clamp, not a validation rule — a bigger pool is served, not rejected |
| `PORT` | `8002` | |

## The two things worth knowing before changing anything

**Feature order is a contract.** `feature_store.schema.FEATURES` defines the column order of
every matrix LightGBM ever sees. `train_lightgbm.py` records the names in the model signature
and the service refuses to start if they disagree with the deployed code. Appending is safe;
renaming or reordering invalidates every trained model, which is what the check exists to catch.

**An empty result never overwrites a good artifact.** Every job exits 1 and leaves the previous
data in place. Writing an empty `rec:trending:top100` would drop every user one rung while the
feed kept answering 200 — a failure with no error anywhere.

## Reading a training run

`train_lightgbm` reports the holdout NDCG@20 next to two baselines, and the comparison is the
result — the absolute number is not. Its value depends on group sizes and on how many positives
the negative sampler left behind, so it means nothing alone.

```
holdout NDCG@20 0.7512, MAP@20 0.6653 (train 0.7566) vs baselines: random 0.3659, popularity 0.4554
```

*random* is the floor a ranker must clear to be worth deploying at all. *popularity* is what the
`trending` rung already achieves for free, with no model — a ranker that only matches it has
learned the one feature the fallback already has. A large gap between train and holdout is the
usual overfit signal.

`MAP@20` at exactly `1.0` is not a triumph, it is a symptom: every item in every group was
relevant, so any ordering scored perfectly. The job refuses a set with no negatives for that
reason.

## Tests

```bash
poetry run pytest        # ~207 tests
poetry run ruff check .
```

No Redis, no ClickHouse, no network, no model download, no MLflow server. SBERT is faked with a
seeded hashing encoder, LightGBM with a booster that only implements `predict`, and Redis with a
hand-rolled dict — the same approach `explAIned-faiss` takes.

One test reaches outside: `test_encoding_matches_explained_faiss_byte_for_byte` imports the
sibling repository's `codec.py` and compares output byte for byte. It skips if that checkout is
missing, and `FAISS_REPO` overrides the path. Since `codec.py` here is a deliberate copy, that
test is the only thing keeping the two honest.

## What is a fixture and what is not

`seed_dev_corpus.py`, `simulate_behavior.py` and the `--encoder hashing` flag exist so the
pipeline can be exercised end to end without model weights or a real user base. They generate
**structured** data on purpose — users have hidden topic preferences a model is supposed to
recover — because uniformly random interactions make NDCG noise around chance, and a broken
ranker indistinguishable from a working one.

None of it is data. Numbers measured on it describe the generator as much as the model.
