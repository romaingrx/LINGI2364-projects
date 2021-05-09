#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 09, 15:14:03
@last modified : 2021 Apr 16, 18:57:40
"""


from collections import defaultdict
from threading import Thread


class PrefixSpan:
    def __init__(self, dataset):
        self._dataset = dataset
        self._k = -1
        self._score_counts = defaultdict(int)
        self._results = []
        self._threads = []
        N, P = (
            self._dataset.n_negative,
            self._dataset.n_positive,
        )  # Get the total number of negative and positive transactions
        self.positive_part = N / (N + P) ** 2
        self.negative_part = P / (N + P) ** 2

    @property
    def least_best_score(self):
        """least_best_support.
        :return: the value of the least best support
        """
        return self._results[0][0] if self._results else 0

    @property
    def current_number_of_k(self):
        """current_number_of_k.
        :return: the number of current supports
        """
        return len(self._score_counts)

    def __call__(self, k):
        self._k = k

        # Starting entries : all tid with pid set to -1
        starting_entries = [(i, -1) for i in range(len(self._dataset._db))]
        starting_pattern = []  # Void pattern

        # Start the main process with void pattern
        self._main_recursive(starting_pattern, starting_entries)

        # Wait all threads
        for thread in self._threads:
            thread.join()

        # Output the results on the stdout
        IO.to_stdout(self._dataset, self._results)

        return self._results

    def next_entries(self, entries):
        # Get the next sequences based on the actual entries (pid+1: for each previous entries)
        next_sequences = (self._dataset._db[tid][pid + 1 :] for tid, pid in entries)

        next_entries_dict = defaultdict(list)

        # Compute the next lists for each `next_sequences`
        for idx, sequence in enumerate(next_sequences):
            # Current entries
            tid, pid = entries[idx]

            # Compute the list of elem as the actual transaction identifier and the next pattern identifier
            for next_pid, item in enumerate(sequence, start=(pid + 1)):
                L = next_entries_dict[item]

                if L and L[-1][0] == tid:
                    continue

                L.append((tid, next_pid))

        return next_entries_dict

    def _update_results(self, pattern, matches, score):
        from bisect import insort

        # If the result is already contained in the results OR \
        #        we already have `k` support values AND the support is lower than the least best support we already have
        if (
            (score, pattern, matches) in self._results
            or self.current_number_of_k == self._k
            and score < self.least_best_score
        ):
            return

        # Increment the number of results for this support
        self._score_counts[score] += 1

        # If we already have `k` score values but the score is greater than the least best score we already have,
        # we have to delete the current least best score results
        if self.current_number_of_k == self._k + 1:
            number_of_least_best_results = self._score_counts[
                self.least_best_score
            ]  # Get the number of results for the least best score values
            del self._score_counts[self.least_best_score]  # Delete this score
            self._results = self._results[
                number_of_least_best_results:
            ]  # Keep all results but the least best results

        # Add the new result when keeping the order of the results
        insort(self._results, (score, pattern, matches))

    def _main_recursive(self, pattern, matches, support=0):
        # If we have a pattern, update the results list
        if len(pattern) > 0:
            self._update_results(pattern, matches, support)

        # Get the next entries (tid, pid)
        new_entries = self.next_entries(matches)
        # Add the score and support for each entry
        # new_entries_score_support = list(Pool().map(self.next_entries_worker, new_entries.items()))
        new_entries_score_support = [
            (item, matches, *self._get_score_key(matches))
            for item, matches in new_entries.items()
        ]
        # Sort the entries by score
        new_entries_score_support.sort(key=lambda x: x[2], reverse=True)

        # For each item and list new_matches, we extend the tree by recursively call the main function on the new patterns
        for new_item, new_matches, score, support in new_entries_score_support:

            # Already have results for this k and the support is lower than the kth best, prune.
            if self.current_number_of_k == self._k and score < self.least_best_score:
                break

            # Current pattern + new item
            new_pattern = pattern + [new_item]

            # Call the main function on the new pattern
            next_thread = Thread(
                target=self._main_recursive,
                args=(new_pattern, new_matches, support),
                daemon=True,
            )
            self._threads.append(next_thread)
            next_thread.start()

    def _get_score_key(self, match):
        """_get_score_key.
        The abstract method that return the upper bound and the actual score of `matches`
        """
        abstract


class SumSupPrefixSpan(PrefixSpan):
    def _get_score_key(self, matches):
        # the sum of both negative and positive supports for the upper bound and key
        return len(matches), len(matches)


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
            "-c",
            "--cprofile",
            help="Run the cprofiler",
            type=lambda x: (str(x).lower() in ["true", "1"]),
            default=False,
        )

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
