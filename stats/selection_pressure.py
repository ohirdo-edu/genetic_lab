from collections import Counter

import numpy as np
import pytest
import scipy
import math


def fisher_stat(ids_before_selection: list[int], ids_after_selection: list[int], fitness: list[float]) -> float:
    assert len(ids_before_selection) == len(ids_after_selection)
    assert len(ids_before_selection) == len(fitness)

    after_selection_counts = Counter(ids_after_selection)
    offspring_counts = [after_selection_counts[id_before] for id_before in ids_before_selection]
    assert sum(offspring_counts) == len(fitness)

    t_median = np.median(fitness)
    c_median = np.median(offspring_counts)

    contingency = [[
        sum(1 for t, c in zip(fitness, offspring_counts) if t <= t_median and c <= c_median),
        sum(1 for t, c in zip(fitness, offspring_counts) if t > t_median and c <= c_median),
    ], [
        sum(1 for t, c in zip(fitness, offspring_counts) if t <= t_median and c > c_median),
        sum(1 for t, c in zip(fitness, offspring_counts) if t > t_median and c > c_median)
    ]]

    result = scipy.stats.fisher_exact(contingency, alternative='greater')

    return -np.log10(result.pvalue)


def kendall_stat(ids_before_selection: list[int], ids_after_selection: list[int], fitness: list[float]) -> float:
    assert len(ids_before_selection) == len(ids_after_selection)
    assert len(ids_before_selection) == len(fitness)

    after_selection_counts = Counter(ids_after_selection)
    offspring_counts = [after_selection_counts[id_before] for id_before in ids_before_selection]

    # result = scipy.stats.kendalltau(fitness, offspring_counts, alternative='greater')
    # assert not math.isnan(result.statistic)
    # if math.isnan(result.statistic):
    #     pass

    n = len(fitness)
    n_concordant = 0
    n_discordant = 0
    n_t = 0
    n_c = 0
    for i in range(n):
        t_i = fitness[i]
        c_i = offspring_counts[i]
        for j in range(i + 1, n):
            t_j = fitness[j]
            c_j = offspring_counts[j]
            if (t_i > t_j and c_i > c_j) or (t_i < t_j and c_i < c_j):
                n_concordant += 1
            if (t_i > t_j and c_i < c_j) or (t_i < t_j and c_i > c_j):
                n_discordant += 1
            if t_i == t_j:
                n_t += 1
            if c_i == c_j:
                n_c += 1

    n_choose_2 = math.comb(n, 2)
    denominator = (n_choose_2 - n_t) * (n_choose_2 - n_c)
    if denominator == 0:
        return 0

    return (n_concordant - n_discordant) / np.sqrt(denominator)


def test_fisher_stat():
    expected = 0.58
    actual = fisher_stat(
        ids_before_selection=list(range(1, 10 + 1)),
        ids_after_selection=[3, 5, 5, 6, 8, 8, 9, 9, 10, 10],
        fitness=[0, 1, 1, 2, 3, 4, 5, 5, 7, 9],
    )
    assert expected == pytest.approx(actual, abs=0.005)


def test_kendall_stat():
    expected = 0.54
    actual = kendall_stat(
        ids_before_selection=list(range(1, 10 + 1)),
        ids_after_selection=[3, 5, 5, 6, 8, 8, 9, 9, 10, 10],
        fitness=[0, 1, 1, 2, 3, 4, 5, 5, 7, 9],
    )
    assert expected == pytest.approx(actual, abs=0.005)
