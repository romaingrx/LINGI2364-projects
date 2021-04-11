#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 09, 16:22:38
@last modified : 2021 Apr 11, 18:28:32
"""


import importlib
from utils import IO, Dataset, get_negative_positive_support
core_prefixspan = importlib.import_module("00-core_prefixspan")
PrefixSpan = core_prefixspan.PrefixSpan


class WraccPrefixSpan(PrefixSpan):

    def _get_score_key(self, matches, return_supports=False):
        n, p = get_negative_positive_support(
            self._dataset, matches
        )  # Get negative and positive support of `matches`

        positive_part = p * self.positive_part
        negative_part = n * self.negative_part
        wracc = positive_part - negative_part

        # Get at least one occurence in the dataset
        if p == 0:
            positive_part = -negative_part / n

        if return_supports:
            return round(positive_part, 5), round(wracc, 5), p, n
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
