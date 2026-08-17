import numpy as np

from explained_ml.ranking_metrics import average_precision_at_k, grouped, ndcg_at_k


def test_a_perfect_ranking_scores_one():
    assert ndcg_at_k([3, 2, 1], [9.0, 5.0, 1.0], k=3) == 1.0


def test_a_reversed_ranking_scores_below_one():
    assert ndcg_at_k([3, 2, 1], [1.0, 5.0, 9.0], k=3) < 1.0


def test_ndcg_matches_a_hand_computation():
    # labels [0, 1] ranked correctly by score: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    value = ndcg_at_k([0, 1], [0.1, 0.9], k=2)

    assert np.isclose(value, 1.0)

    # Now ranked wrongly: the relevant item lands second.
    wrong = ndcg_at_k([0, 1], [0.9, 0.1], k=2)

    assert np.isclose(wrong, 1.0 / np.log2(3))


def test_a_group_with_no_positives_scores_zero_not_one():
    # An all-negative group has no attainable ranking. Scoring it 1.0 (vacuously perfect)
    # would let a sampler that produced empty groups inflate the reported metric.
    assert ndcg_at_k([0, 0, 0], [1.0, 2.0, 3.0], k=3) == 0.0


def test_an_empty_group_scores_zero():
    assert ndcg_at_k([], [], k=20) == 0.0
    assert average_precision_at_k([], [], k=20) == 0.0


def test_cutoff_is_respected():
    labels = [0] * 20 + [3]
    scores = list(range(21))[::-1]

    assert ndcg_at_k(labels, scores, k=20) == 0.0
    assert ndcg_at_k(labels, scores, k=21) > 0.0


def test_ties_are_broken_by_input_order_so_results_reproduce():
    first = ndcg_at_k([1, 0], [0.5, 0.5], k=2)
    second = ndcg_at_k([1, 0], [0.5, 0.5], k=2)

    assert first == second == 1.0


def test_average_precision_rewards_early_hits():
    early = average_precision_at_k([1, 0, 0], [3.0, 2.0, 1.0], k=3)
    late = average_precision_at_k([0, 0, 1], [3.0, 2.0, 1.0], k=3)

    assert early > late


def test_average_precision_of_a_perfect_ranking_is_one():
    assert average_precision_at_k([1, 1, 0], [3.0, 2.0, 1.0], k=3) == 1.0


def test_grouped_averages_per_group_not_per_row():
    # One user with many impressions must not outvote several users with few.
    labels = [1, 0] + [0, 1]
    scores = [9.0, 1.0] + [9.0, 1.0]

    result = grouped(labels, scores, groups=[2, 2], k=20)

    assert 0.0 < result["ndcg_at_20"] < 1.0
    assert set(result) == {"ndcg_at_20", "map_at_20"}


def test_grouped_handles_ragged_groups():
    result = grouped([1, 0, 0, 1], [2.0, 1.0, 1.0, 2.0], groups=[3, 1], k=20)

    assert result["ndcg_at_20"] > 0.0
