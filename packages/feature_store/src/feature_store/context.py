"""What a feature is allowed to see.

Both callers — `train_lightgbm.py` offline and the ranking service online — build these same
two objects and hand them to `compute`. That is the whole anti-skew mechanism: if a feature
cannot be expressed from a context, it cannot silently be computed one way in training and
another way in serving.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class UserContext:
    user_id: str
    # Flat name -> value map, exactly as materialised into `rec:user_features:{uid}`.
    features: dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def get(self, name: str, default: float = 0.0) -> float:
        return self.features.get(name, default)


@dataclass(frozen=True, slots=True)
class ArticleContext:
    article_id: str
    author_id: str = ""
    features: dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def get(self, name: str, default: float = 0.0) -> float:
        return self.features.get(name, default)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Everything that is neither user nor article — today just the clock.

    Passed explicitly rather than read from `datetime.now()` inside a feature so that training
    on historical rows produces the values that were true then, not the values true now.
    """

    now: datetime
