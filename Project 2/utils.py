#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 10, 11:50:37
@last modified : 2021 Apr 11, 12:56:57
"""

def starfilter(func, iterable):
    star_func = lambda arg:func(*arg)
    return filter(star_func, iterable)

def get_negative_positive_support(dataset, matches):
    from bisect import bisect_left

    positive_support = bisect_left(
        matches, (dataset.n_positive, -1)
    )  # Get the index of the maximum positive index in the matches
    negative_support = len(matches) - positive_support
    return negative_support, positive_support

class Dataset:
    def __init__(self, negative_file, positive_file):
        self.transactions = []
        self.unique_items = set()

        def append_transactions(filepath):
            """append_transactions.
            Append the transactions contained in the file `filepath`

            :param filepath: a path to the file
            """
            starting_length = len(
                self.transactions
            )  # Used to get the number of transactions at the end of the function
            if not self.transactions:  # If we don't have any transactions
                self.transactions.append([])

            with open(filepath, "r") as fd:
                lines = [line.strip() for line in fd]
                for line in lines:
                    if line:
                        item = line.split(" ")[0]
                        self.unique_items.add(item)  # Add unique items in the set
                        self.transactions[-1].append(item)
                    elif self.transactions[-1] != []:
                        self.transactions.append([])

            del self.transactions[-1]  # Because last appended is a void list
            return (
                len(self.transactions) - starting_length
            )  # Return the number of transactions

        self.n_positive = append_transactions(positive_file)
        self.n_negative = append_transactions(negative_file)

        self._item_to_int = {
            item: idx for idx, item in enumerate(self.unique_items)
        }  # Mapping item -> indexes
        self._int_to_item = {
            v: k for k, v in self._item_to_int.items()
        }  # Invert the key value dict to value key

        self._db = [
            [self._item_to_int[item] for item in transaction]
            for transaction in self.transactions
        ]  # Create the database with the transaction items expressed as integer

        # Not needed anymore
        del self._item_to_int

class IO:
    @staticmethod
    def output(dataset, results, fd):
        for support, pattern, matches, *_ in results:
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
                "-c", "--cprofile", help="Run the cprofiler", type=lambda x: (str(x).lower() in ['true', '1']), default=False
        )

        args = parser.parse_args()
        return args
