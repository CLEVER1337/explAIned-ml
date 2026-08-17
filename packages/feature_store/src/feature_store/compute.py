"""Turning contexts into a matrix. No IO lives here, and no clock either.

`train_lightgbm.py` and the ranking service both call `build_matrix`. That is the only reason
this package exists as a package: the arithmetic has exactly one implementation.
"""

from collections.abc import Sequence

import numpy as np

from .context import ArticleContext, RequestContext, UserContext
from .schema import FEATURES


def build_row(user: UserContext, article: ArticleContext, request: RequestContext) -> np.ndarray:
    """One candidate -> one row, in `FEATURES` order."""
    return np.fromiter(
        (spec.fn(user, article, request) for spec in FEATURES),
        dtype=np.float32,
        count=len(FEATURES),
    )


def build_matrix(
    user: UserContext, articles: Sequence[ArticleContext], request: RequestContext
) -> np.ndarray:
    """`(len(articles), len(FEATURES))`, ready for `booster.predict`.

    Row order matches `articles` so the caller can zip scores back onto ids without a join.
    """
    if not articles:
        return np.zeros((0, len(FEATURES)), dtype=np.float32)

    matrix = np.empty((len(articles), len(FEATURES)), dtype=np.float32)
    for index, article in enumerate(articles):
        matrix[index] = build_row(user, article, request)

    # A NaN reaching LightGBM is silently treated as a missing value, which hides the bug that
    # produced it. Features are supposed to be total functions; make a violation visible.
    if not np.isfinite(matrix).all():
        bad = [FEATURES[c].name for c in np.unique(np.argwhere(~np.isfinite(matrix))[:, 1])]
        raise ValueError(f"non-finite feature values produced by: {', '.join(bad)}")

    return matrix


def explain(
    user: UserContext, article: ArticleContext, request: RequestContext
) -> dict[str, float]:
    """Named values for one candidate — what `--explain` and debugging actually need."""
    row = build_row(user, article, request)
    return {spec.name: float(value) for spec, value in zip(FEATURES, row, strict=True)}
