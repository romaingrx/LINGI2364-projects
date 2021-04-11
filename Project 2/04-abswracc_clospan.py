#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 11, 15:31:00
@last modified : 2021 Apr 11, 16:06:08
"""

from utils import IO, get_negative_positive_support
from importlib import import_module
wracc_clospan = import_module("03-wracc_clospan")
Dataset, WraccCloSpan = wracc_clospan.Dataset, wracc_clospan.WraccCloSpan

class AbsWraccCloSpan(WraccCloSpan):
    def _get_score_key(self, matches, return_supports=False):
        n, p = get_negative_positive_support(
            self._dataset, matches
        )  # Get negative and positive support of `matches`

        positive_part = p * self.positive_part
        negative_part = n * self.negative_part
        abswracc = abs(positive_part - negative_part)

        # Get at least one occurence in the dataset
        key = positive_part if p > 0 else -self.negative_part
        key = max(key, negative_part if n > 0 else -self.positive_part)

        if return_supports:
            return round(key, 5), round(abswracc, 5), p, n
        return round(key, 5), round(abswracc, 5)


if __name__ == "__main__":
    import cProfile, pstats

    args = IO.from_stdin()
    ds = Dataset(args.negative_filepath, args.positive_filepath)
    algo = AbsWraccCloSpan(ds)
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = algo(args.k)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()
    else:
        results = algo(args.k)
