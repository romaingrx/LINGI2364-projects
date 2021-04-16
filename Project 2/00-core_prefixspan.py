#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 11, 18:06:20
@last modified : 2021 Apr 16, 16:42:49
"""

from utils import IO

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
