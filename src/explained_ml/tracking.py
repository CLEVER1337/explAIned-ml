"""MLflow, kept at arm's length.

Only two things ever touch it: the training jobs write runs, and the ranking service resolves
one alias at startup. Everything else about the recommendation loop lives in Redis, on purpose
— embeddings, ALS candidate lists and trending are large, regenerable, and read on a hot path.
MLflow holds what is *not* regenerable: the weights, the metrics that justified promoting them,
and the feature signature that says what those weights expect to be fed.

The backend is SQLite rather than `file:./mlruns`. The filesystem tracking and registry stores
were deprecated in February 2026 and warn on every call; SQLite is still a single file and no
daemon, which was the reason to pick the file store in the first place.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """No model version is published under the alias we were told to serve."""


def configure(settings: Settings) -> None:
    import mlflow

    _ensure_local_dirs(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)


def _ensure_local_dirs(settings: Settings) -> None:
    """SQLite will not create the directory holding its file, and MLflow's error for that is
    an opaque OperationalError."""
    for uri in (settings.mlflow_tracking_uri, settings.mlflow_registry_uri):
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    Path(settings.mlflow_artifact_root).mkdir(parents=True, exist_ok=True)


@contextmanager
def run(settings: Settings, name: str):
    import mlflow

    configure(settings)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name=name) as active:
        yield active


def resolve_alias(settings: Settings) -> tuple[str, str]:
    """`(model_uri, version)` for the configured alias.

    Deliberately by alias and not by path: promoting a model must be a registry operation, not
    a redeploy. `ModelUnavailable` is a *normal* startup state before the first promotion — it
    is not the same thing as a model that disagrees with the code.
    """
    from mlflow.tracking import MlflowClient

    configure(settings)
    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias(
            settings.mlflow_model_name, settings.mlflow_model_alias
        )
    except Exception as exc:  # mlflow raises a family of errors here, all meaning "not there"
        raise ModelUnavailable(
            f"no version aliased @{settings.mlflow_model_alias} for "
            f"{settings.mlflow_model_name}: {exc}"
        ) from exc

    return f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}", str(version.version)


def promote(settings: Settings, version: str) -> None:
    from mlflow.tracking import MlflowClient

    configure(settings)
    MlflowClient().set_registered_model_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias, version
    )
    logger.info(
        "moved @%s to %s version %s", settings.mlflow_model_alias, settings.mlflow_model_name, version
    )


def champion_metric(settings: Settings, metric: str) -> float | None:
    """The metric the currently-promoted model scored, so promotion can be a comparison rather
    than a habit."""
    from mlflow.tracking import MlflowClient

    configure(settings)
    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias(
            settings.mlflow_model_name, settings.mlflow_model_alias
        )
        run_data = client.get_run(version.run_id)
    except Exception:
        return None

    value = run_data.data.metrics.get(metric)
    return float(value) if value is not None else None
