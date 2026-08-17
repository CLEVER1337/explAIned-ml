"""Feature definitions shared by `train_lightgbm.py` and the ranking service (:8002).

Deliberately IO-free: numpy in, numpy out. Fetching belongs to `explained_ml` — this package
only decides what a feature *is*, so that training and serving cannot disagree about it.
"""

from .compute import build_matrix, build_row, explain
from .context import ArticleContext, RequestContext, UserContext
from .schema import FEATURES, FeatureSpec, feature_names
from .signature import FeatureSchemaMismatch, assert_matches, schema_version

__all__ = [
    "FEATURES",
    "ArticleContext",
    "FeatureSchemaMismatch",
    "FeatureSpec",
    "RequestContext",
    "UserContext",
    "assert_matches",
    "build_matrix",
    "build_row",
    "explain",
    "feature_names",
    "schema_version",
]
