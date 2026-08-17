"""The loaded model, and the rules for replacing it.

Shaped after `explained_faiss.service.state`: an immutable bundle behind a lock, swapped by
reference so in-flight requests keep the model they started with, and **a failed reload never
drops the model that is currently serving**. A stale ranker beats no ranker — the orchestrator
treats "no ranker" as a drop to `unranked` for every user at once.

The startup check is the asymmetry worth reading twice: *no model yet* is a normal state that
lets the service start and answer 503, while *a model that disagrees with this code's feature
list* is fatal. The first happens before the first training run; the second means every score
would be computed from columns shifted against their meaning, silently.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from feature_store.signature import FeatureSchemaMismatch, assert_matches, schema_version

from ..metrics import MODEL_FEATURES, MODEL_LOADED, MODEL_RELOAD_TOTAL
from ..tracking import ModelUnavailable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    booster: object
    feature_names: tuple[str, ...]
    name: str
    version: str
    schema_version: str
    loaded_at: datetime

    def predict(self, matrix):
        return self.booster.predict(matrix)


@dataclass(frozen=True, slots=True)
class ReloadResult:
    result: str  # loaded | unchanged | missing | failed | signature_mismatch
    version: str | None = None
    error: str | None = None


class ModelState:
    def __init__(self, loader) -> None:
        self._loader = loader
        self._bundle: ModelBundle | None = None
        self._lock = asyncio.Lock()

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle

    async def reload(self, force: bool = False) -> ReloadResult:
        async with self._lock:
            try:
                version = await asyncio.to_thread(self._loader.current_version)
            except ModelUnavailable as exc:
                MODEL_RELOAD_TOTAL.labels(result="missing").inc()
                return ReloadResult("missing", error=str(exc))
            except Exception as exc:  # noqa: BLE001 - registry down is not our crash
                MODEL_RELOAD_TOTAL.labels(result="failed").inc()
                logger.warning("could not reach the model registry: %s", exc)
                return ReloadResult("failed", error=str(exc))

            if not force and self._bundle is not None and self._bundle.version == version:
                MODEL_RELOAD_TOTAL.labels(result="unchanged").inc()
                return ReloadResult("unchanged", version=version)

            try:
                bundle = await asyncio.to_thread(self._loader.load)
            except FeatureSchemaMismatch as exc:
                MODEL_RELOAD_TOTAL.labels(result="signature_mismatch").inc()
                logger.error("refusing the model: %s", exc)
                return ReloadResult("signature_mismatch", version=version, error=str(exc))
            except ModelUnavailable as exc:
                MODEL_RELOAD_TOTAL.labels(result="missing").inc()
                return ReloadResult("missing", error=str(exc))
            except Exception as exc:  # noqa: BLE001
                MODEL_RELOAD_TOTAL.labels(result="failed").inc()
                logger.warning("model load failed, keeping the current one: %s", exc)
                return ReloadResult("failed", version=version, error=str(exc))

            self._bundle = bundle
            MODEL_LOADED.set(1)
            MODEL_FEATURES.set(len(bundle.feature_names))
            MODEL_RELOAD_TOTAL.labels(result="loaded").inc()
            logger.info(
                "serving %s version %s (%d features, schema %s)",
                bundle.name,
                bundle.version,
                len(bundle.feature_names),
                bundle.schema_version,
            )

            return ReloadResult("loaded", version=bundle.version)

    def describe(self) -> dict:
        if self._bundle is None:
            return {
                "model_loaded": False,
                "model_name": None,
                "model_version": None,
                "schema_version": schema_version(),
                "features": 0,
                "loaded_at": None,
            }

        return {
            "model_loaded": True,
            "model_name": self._bundle.name,
            "model_version": self._bundle.version,
            "schema_version": self._bundle.schema_version,
            "features": len(self._bundle.feature_names),
            "loaded_at": self._bundle.loaded_at.isoformat(),
        }


class MlflowModelLoader:
    """Resolves the configured alias and refuses anything whose signature disagrees with us."""

    def __init__(self, settings) -> None:
        self._settings = settings

    def current_version(self) -> str:
        from ..tracking import resolve_alias

        _uri, version = resolve_alias(self._settings)
        return version

    def load(self) -> ModelBundle:
        import mlflow

        from ..tracking import resolve_alias

        uri, version = resolve_alias(self._settings)

        model = mlflow.lightgbm.load_model(uri)
        names = _signature_features(uri)

        # The whole reason the service refuses to start on a mismatch: nothing else compares
        # what train_lightgbm.py recorded against what this feature_store produces.
        assert_matches(names)

        return ModelBundle(
            booster=model,
            feature_names=tuple(names),
            name=self._settings.mlflow_model_name,
            version=version,
            schema_version=schema_version(),
            loaded_at=datetime.now(UTC),
        )


def _signature_features(uri: str) -> list[str]:
    import mlflow

    info = mlflow.models.get_model_info(uri)
    signature = getattr(info, "signature", None)
    if signature is None or signature.inputs is None:
        raise FeatureSchemaMismatch(
            "the registered model carries no signature, so its feature list cannot be checked; "
            "retrain with mlflow.models.infer_signature"
        )

    return [spec.name for spec in signature.inputs.inputs]
