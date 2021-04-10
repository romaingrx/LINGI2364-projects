#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 10, 11:50:37
@last modified : 2021 Apr 11, 00:53:34
"""


def get_negative_positive_support(dataset, matches):
    from bisect import bisect_left

    positive_support = bisect_left(
        matches, (dataset.n_positive, -1)
    )  # Get the index of the maximum positive index in the matches
    negative_support = len(matches) - positive_support
    return negative_support, positive_support


class IO:
    @staticmethod
    def output(dataset, results, fd):
        for support, pattern, matches in results:
            n, p = get_negative_positive_support(dataset, matches)
            items = [dataset._int_to_item[integer] for integer in pattern]

            fd.write(f'[{", ".join(items)}] {p} {n} {support}\n')

    @staticmethod
    def to_stdout(dataset, results):
        import sys

        IO.output(dataset, results, sys.stdout)

    @staticmethod
    def to_file(dataset, results, filepath):
        fd = open(filepath, "w+")
        IO.output(dataset, results, fd)

    @staticmethod
    def from_stdin():
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument(
            "positive_filepath", help="Path to the positive file", type=str
        )
        parser.add_argument(
            "negative_filepath", help="Path to the negative file", type=str
        )
        parser.add_argument("k", help="The number of top sequential patterns", type=int)
        parser.add_argument(
            "-c", "--cprofile", help="Run the cprofiler", type=bool, default=False
        )

        args = parser.parse_args()
        return args
