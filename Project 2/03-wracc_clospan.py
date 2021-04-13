#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 11, 12:14:03
@last modified : 2021 Apr 11, 18:25:28
"""

from importlib import import_module
from utils import IO, Dataset

core_clospan = import_module("00-core_clospan")
CloSpan = core_clospan.CloSpan

wracc_prefixspan = import_module("02-wracc_prefixspan")
WraccPrefixSpan = wracc_prefixspan.WraccPrefixSpan


class WraccCloSPan(CloSpan, WraccPrefixSpan):
    # Get all methods from CloSpan + _get_score_key from WraccPrefixSpan
    def __init__(self, ds):
        super(WraccCloSPan, self).__init__(ds)


if __name__ == "__main__":
    import cProfile, pstats

    args = IO.from_stdin()
    ds = Dataset(args.negative_filepath, args.positive_filepath)
    algo = WraccCloSPan(ds)
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = algo(args.k)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()
    else:
        results = algo(args.k)
