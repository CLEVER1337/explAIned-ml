"""The train/serve skew guard.

`train_lightgbm.py` records `feature_names()` in the model's MLflow signature; the ranking
service calls `assert_matches` against the same list at startup and refuses to serve on a
mismatch. Nothing else checks this boundary — not the compiler, not code review, not even
living in one repository, because a model trained at commit A can be served by code at
commit B.

The failure this prevents is the expensive kind: a renamed or reordered feature does not
crash, it silently feeds column 7's values into column 8's meaning and the model quietly
gets worse.
"""

import hashlib
from collections.abc import Sequence

from .schema import feature_names


class FeatureSchemaMismatch(RuntimeError):
    """The loaded model expects a different feature list than this code produces."""


def schema_version() -> str:
    """Short, stable fingerprint of the ordered feature list. Logged and exported as a label."""
    joined = "|".join(feature_names()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]


def assert_matches(model_features: Sequence[str]) -> None:
    expected = feature_names()
    actual = list(model_features)

    if actual == expected:
        return

    raise FeatureSchemaMismatch(_describe(expected, actual))


def _describe(expected: list[str], actual: list[str]) -> str:
    if len(expected) != len(actual):
        head = f"model expects {len(actual)} features, this feature_store produces {len(expected)}"
    else:
        head = "feature lists differ in order or naming"

    detail = ""
    for index, (want, got) in enumerate(zip(expected, actual, strict=False)):
        if want != got:
            detail = f"; first difference at index {index}: model has {got!r}, code has {want!r}"
            break

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        detail += f"; missing from model: {', '.join(missing)}"
    if extra:
        detail += f"; unknown to code: {', '.join(extra)}"

    return f"{head}{detail}. Retrain, or check out the commit the model was trained from."
