#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 11, 16:09:36
@last modified : 2021 Apr 16, 16:22:50
"""

from math import log2
from functools import lru_cache

from utils import IO, Dataset, get_negative_positive_support
from importlib import import_module

core_clospan = import_module("00-core_clospan")
CloSpan = core_clospan.CloSpan


class InfoGainCloSpan(CloSpan):
    def __init__(self, ds):
        super(InfoGainCloSpan, self).__init__(ds)
        P = self._dataset.n_positive
        N = self._dataset.n_negative

        # Compute the information gain value for each (p, n) tuple
        self._information_gain_values = [
            [round(self._information_gain(p, n), 5) for n in range(N + 1)]
            for p in range(P + 1)
        ]

        # Compute the upper bound for each (p, n) tuple
        self._key_values = [l.copy() for l in self._information_gain_values]
        # Cumulative maximum
        for p in range(P + 1):
            for n in range(N + 1):
                self._key_values[p][n] = max(
                    self._information_gain_values[p][n],
                    self._key_values[max(0, p - 1)][n],
                    self._key_values[p][max(0, n - 1)],
                )

    def _information_gain(self, p, n):
        def entropy(x):
            # Take care of the bounds
            if x <= 0 or x >= 1:
                return 0.0
            return -x * log2(x) - (1 - x) * log2(1 - x)

        P = self._dataset.n_positive
        N = self._dataset.n_negative

        # Compute the info gain and consider non definite members as .0
        value = 0.0
        if P + N:
            value += entropy(P / (P + N))

            if p + n:
                value -= (p + n) / (P + N) * entropy(p / (p + n))

            if P + N - p - n:
                value -= (P + N - p - n) / (P + N) * entropy((P - p) / (P + N - p - n))

        return value

    def _get_score_key(self, matches, return_supports=False):
        n, p = get_negative_positive_support(self._dataset, matches)
        if return_supports:
            return self._key_values[p][n], self._information_gain_values[p][n], p, n
        return self._key_values[p][n], self._information_gain_values[p][n]


if __name__ == "__main__":
    import cProfile, pstats

    args = IO.from_stdin()
    ds = Dataset(args.negative_filepath, args.positive_filepath)
    algo = InfoGainCloSpan(ds)
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = algo(args.k)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()
    else:
        results = algo(args.k)
