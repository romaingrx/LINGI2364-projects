#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 09, 15:14:03
@last modified : 2021 Apr 10, 20:01:56
"""

from utils import IO

from collections import defaultdict


class IO:
    @staticmethod
    def output(dataset, results, fd):
        def get_negative_positive_support(matches):
            from bisect import bisect_left

            positive_support = bisect_left(
                matches, (dataset.n_positive, -1)
            )  # Get the index of the maximum positive index in the matches
            negative_support = len(matches) - positive_support
            return negative_support, positive_support

        for support, pattern, matches in results:
            n, p = get_negative_positive_support(matches)
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

        args = parser.parse_args()
        return args


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


class PrefixSpan:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._k = -1
        self._support_counts = defaultdict(int)
        self._results = []

    @property
    def least_best_support(self):
        """least_best_support.
        :return: the value of the least best support
        """
        return self._results[0][0] if self._results else 0

    @property
    def current_number_of_k(self):
        """current_number_of_k.
        :return: the number of current supports
        """
        return len(self._support_counts)

    def __call__(self, k):
        # for l in range(1, k+1):
        #     self._k = l
        self._k = k

        # Starting entries : all tid with pid set to -1
        starting_entries = [(i, -1) for i in range(len(self._dataset._db))]
        starting_key = 0
        starting_pattern = []  # Void pattern
        self._main_recursive(starting_pattern, starting_entries, starting_key)

        IO.to_stdout(self._dataset, self._results)

        return self._results

    def next_entries(self, entries):
        next_sequences = (
            self._dataset._db[k][last_position + 1 :] for k, last_position in entries
        )

        next_entries_dict = defaultdict(list)

        for idx, sequence in enumerate(next_sequences):
            tid, pid = entries[idx]

            for next_pid, item in enumerate(sequence, start=(pid + 1)):
                L = next_entries_dict[item]

                if L and L[-1][0] == tid:
                    continue

                L.append((tid, next_pid))

        return next_entries_dict

    def _get_score_support(self, match):
        return len(match), len(match)

    def _update_results(self, pattern, matches, support):
        from bisect import insort

        # If the result is already contained in the results OR \
        #        we already have `k` support values AND the support is lower than the least best support we already have
        if (
            (support, pattern, matches) in self._results
            or self.current_number_of_k == self._k
            and support < self.least_best_support
        ):
            return

        # Increment the number of results for this support
        self._support_counts[support] += 1

        # If we already have `k` support values but the support is greater than the least best support we already have,
        # we have to delete the current least best support results
        if self.current_number_of_k == self._k + 1:
            number_of_least_best_results = self._support_counts[
                self.least_best_support
            ]  # Get the number of results for the least best support values
            del self._support_counts[self.least_best_support]  # Delete this support
            self._results = self._results[
                number_of_least_best_results:
            ]  # Keep all results but the least best results

        # Add the new result when keeping the order of the results
        insort(self._results, (support, pattern, matches))

    def _main_recursive(self, pattern, matches, support):
        # If we have a pattern, update the results list
        if len(pattern) > 0:
            self._update_results(pattern, matches, support)

        # Get the next entries (tid, pid)
        new_entries = self.next_entries(matches)
        # Add the score and support for each entry
        new_entries_score_support = [
            (item, matches, *self._get_score_support(matches))
            for item, matches in new_entries.items()
        ]
        # Sort the entries by score
        new_entries_score_support.sort(key=lambda x: x[2], reverse=True)

        # For each item and list new_matches, we extend the tree by recursively call the main function on the new patterns
        for new_item, new_matches, score, support in new_entries_score_support:

            # Already have results for this k and the support is lower than the kth best, prune.
            if self.current_number_of_k == self._k and score < self.least_best_support:
                continue

            # Current pattern + new item
            new_pattern = pattern + [new_item]

            # Call the main function on the new pattern
            self._main_recursive(new_pattern, new_matches, support)


if __name__ == "__main__":
    args = IO.from_stdin()
    ds = Dataset(args.negative_filepath, args.positive_filepath)
    algo = PrefixSpan(ds)
    results = algo(args.k)
