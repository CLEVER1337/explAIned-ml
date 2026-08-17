"""NDCG@k and MAP@k over ragged groups.

Hand-rolled rather than sklearn: `ndcg_score` wants a padded 2-D array of equal-length rows,
and a feed's groups are ragged by nature — one user saw 12 candidates on Tuesday and 40 on
Wednesday. Padding them with zeros quietly changes the denominator, which is exactly the number
being reported.

Both metrics are computed per group and averaged, so a user with 200 impressions does not
outvote fifty users with four.
"""

from collections.abc import Sequence

import numpy as np


def _order(scores: np.ndarray) -> np.ndarray:
    """Descending by score, ties broken by original position so results are reproducible."""
    return np.lexsort((np.arange(len(scores)), -scores))


def dcg_at_k(labels: np.ndarray, k: int) -> float:
    top = labels[:k]
    if top.size == 0:
        return 0.0

    gains = (2.0**top) - 1.0
    discounts = np.log2(np.arange(2, top.size + 2))

    return float(np.sum(gains / discounts))


def ndcg_at_k(labels: Sequence[float], scores: Sequence[float], k: int = 20) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0:
        return 0.0

    ideal = dcg_at_k(np.sort(labels)[::-1], k)
    if ideal == 0.0:
        # A group with no positives has no attainable ranking; scoring it 0 would punish the
        # model for the sampler's choices rather than for its own.
        return 0.0

    return dcg_at_k(labels[_order(scores)], k) / ideal


def average_precision_at_k(labels: Sequence[float], scores: Sequence[float], k: int = 20) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0:
        return 0.0

    relevant = labels[_order(scores)][:k] > 0
    total_relevant = int(np.count_nonzero(labels > 0))
    if total_relevant == 0:
        return 0.0

    hits = np.cumsum(relevant)
    precisions = hits / np.arange(1, relevant.size + 1)

    return float(np.sum(precisions * relevant) / min(total_relevant, k))


def grouped(
    labels: Sequence[float], scores: Sequence[float], groups: Sequence[int], k: int = 20
) -> dict[str, float]:
    """`groups` is LightGBM's format: consecutive run lengths, not group ids."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    ndcgs: list[float] = []
    maps: list[float] = []

    start = 0
    for size in groups:
        end = start + int(size)
        ndcgs.append(ndcg_at_k(labels[start:end], scores[start:end], k))
        maps.append(average_precision_at_k(labels[start:end], scores[start:end], k))
        start = end

    return {
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map_at_{k}": float(np.mean(maps)) if maps else 0.0,
    }
