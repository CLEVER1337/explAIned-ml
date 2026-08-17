"""A metric without a baseline is a number, not a result.

`_baselines` is what makes the reported NDCG interpretable: random is the floor a ranker has to
clear to be worth deploying, and popularity is what the `trending` rung already achieves for
free without any model at all.
"""

import numpy as np

from explained_ml.jobs.train_lightgbm import _baselines
from feature_store.schema import feature_names

POPULARITY = feature_names().index("a_clicks_24h_log")


def holdout(rows: list[tuple[float, float]], groups: list[int]):
    """`rows` is (popularity, label) per example."""
    features = np.zeros((len(rows), len(feature_names())), dtype=np.float32)
    for index, (popularity, _label) in enumerate(rows):
        features[index, POPULARITY] = popularity

    labels = np.asarray([label for _p, label in rows], dtype=np.float64)

    return features, labels, groups


def test_both_baselines_are_reported():
    result = _baselines(holdout([(1.0, 1.0), (2.0, 0.0)], [2]))

    assert set(result) == {"baseline_random_ndcg_at_20", "baseline_popularity_ndcg_at_20"}


def test_an_empty_holdout_reports_nothing():
    assert _baselines(holdout([], [])) == {}


def test_the_popularity_baseline_uses_the_popularity_column():
    # Labels agree with popularity, so ordering by it is perfect.
    perfect = _baselines(holdout([(9.0, 3.0), (5.0, 2.0), (1.0, 0.0)], [3]))

    assert perfect["baseline_popularity_ndcg_at_20"] == 1.0

    # Labels are the exact inverse of popularity — the worst this baseline can do.
    inverted = _baselines(holdout([(9.0, 0.0), (5.0, 2.0), (1.0, 3.0)], [3]))

    assert inverted["baseline_popularity_ndcg_at_20"] < perfect["baseline_popularity_ndcg_at_20"]


def test_the_random_baseline_is_reproducible():
    rows = [(float(i), float(i % 2)) for i in range(20)]

    first = _baselines(holdout(rows, [10, 10]))
    second = _baselines(holdout(rows, [10, 10]))

    # A seeded baseline: comparing two training runs must not be confounded by the floor moving.
    assert first["baseline_random_ndcg_at_20"] == second["baseline_random_ndcg_at_20"]


def test_baselines_respect_group_boundaries():
    rows = [(1.0, 1.0), (2.0, 0.0), (3.0, 0.0), (4.0, 1.0)]

    one_group = _baselines(holdout(rows, [4]))
    two_groups = _baselines(holdout(rows, [2, 2]))

    assert one_group["baseline_popularity_ndcg_at_20"] != two_groups["baseline_popularity_ndcg_at_20"]
