# systemd units

Six timers, one per offline job. Deliberately not an orchestrator.

```bash
sudo cp explained-ml-*.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now explained-ml-{trending,embeddings,user-embeddings,features,als,lightgbm}.timer
systemctl list-timers 'explained-ml-*'
```

| Timer | Schedule | Job |
|---|---|---|
| `trending` | every 10 minutes | `update_trending` |
| `embeddings` | hourly | `embeddings` |
| `user-embeddings` | hourly | `build_user_embeddings` |
| `features` | hourly | `build_features` |
| `als` | daily 03:30 | `train_als` |
| `lightgbm` | Sundays 04:30 | `train_lightgbm` (no `--promote`) |

## Why timers and not Prefect or Dagster

`update_trending` is the worst possible candidate for orchestration: one query, one key, every
ten minutes, no upstream dependency. Putting a worker in front of it adds a failure mode that
does not otherwise exist — the worker dies, `rec:trending:top100` goes stale, and every feed
quietly drops to `recent`. That is precisely the state the job exists to prevent.

Introduce Prefect (or Dagster) when there is a real data dependency to express and retries to
manage: `embeddings` → `build_user_embeddings` is one today, and `train_als` on a ClickHouse
that occasionally blinks will be another. Until then a timer is fewer moving parts than the
thing being scheduled.

The rule that survives whichever way that goes: **a job is always runnable on its own**. These
units run `python -m explained_ml.jobs.X` with no wrapper, so a backfill, a debugging run and a
scheduled run are the same command.

## `SuccessExitStatus=0 1`

Exit 1 means "nothing to do" — an empty ClickHouse window, a catalog with no new articles. It is
a normal outcome, not a failure, and the job has deliberately left the previous artifact in
place. Without this line every quiet hour marks the unit failed.

Exit 2 is infrastructure being down and does mark the unit failed, which is correct.

## Ordering

`user-embeddings` and `features` read what `embeddings` writes, but they are scheduled rather
than chained: each is hourly with a randomized delay, and each degrades honestly if its input is
an hour stale. Chaining them with `After=` would mean one slow encode blocks feature
materialization for everybody, to buy freshness that nothing downstream can perceive.

## Environment

Put overrides in `/etc/explained-ml.env` (the `EnvironmentFile=-` prefix makes it optional):

```
REDIS_URL=redis://:gavno@127.0.0.1:6379/0
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_DATABASE=explained
ARTICLES_BASE_URL=http://localhost:5036
MLFLOW_TRACKING_URI=sqlite:////var/lib/explained-ml/mlflow.db
MLFLOW_REGISTRY_URI=sqlite:////var/lib/explained-ml/mlflow.db
```
