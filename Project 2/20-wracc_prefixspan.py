#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 09, 16:22:38
@last modified : 2021 Apr 11, 00:50:31
"""


import importlib
from utils import IO, get_negative_positive_support
sumsup_prefixspan = importlib.import_module("10-sumsup_prefixspan")
PrefixSpan = sumsup_prefixspan.PrefixSpan
Dataset = sumsup_prefixspan.Dataset


class WraccPrefixSpan(PrefixSpan):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N, P = (
            self._dataset.n_negative,
            self._dataset.n_positive,
        )  # Get the total number of negative and positive transactions
        self.positive_part = N / (N + P) ** 2
        self.negative_part = P / (N + P) ** 2
    def _get_score_key(self, matches):
        n, p = get_negative_positive_support(
            self._dataset, matches
        )  # Get negative and positive support of `matches`

        positive_part = p * self.positive_part
        negative_part = n * self.negative_part
        wracc = positive_part - negative_part

        # Get at least one occurence in the dataset
        if p == 0:
            positive_part = -negative_part / n

        return round(positive_part, 5), round(wracc, 5)


if __name__ == "__main__":
    import cProfile, pstats
    args = IO.from_stdin()
    ds = Dataset(args.negative_filepath, args.positive_filepath)
    algo = WraccPrefixSpan(ds)
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = algo(args.k)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()
    else:
        results = algo(args.k)
