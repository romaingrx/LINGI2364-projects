#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 09, 15:14:03
@last modified : 2021 Apr 11, 18:12:55
"""

from importlib import import_module
from utils import IO, Dataset

core_prefixspan = import_module("00-core_prefixspan")
PrefixSpan = core_prefixspan.PrefixSpan


class SumSupPrefixSpan(PrefixSpan):
    def _get_score_key(self, matches):
        return len(matches), len(matches)


if __name__ == "__main__":
    import cProfile, pstats

    args = IO.from_stdin()
    ds = Dataset(args.negative_filepath, args.positive_filepath)
    algo = SumSupPrefixSpan(ds)
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = algo(args.k)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()
    else:
        results = algo(args.k)
