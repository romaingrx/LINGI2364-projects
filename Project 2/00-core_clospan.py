#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 11, 18:14:15
@last modified : 2021 Apr 16, 16:36:43
"""

from importlib import import_module
from itertools import combinations
from threading import Thread
from utils import IO, starfilter

core_prefixspan = import_module("00-core_prefixspan")
PrefixSpan = core_prefixspan.PrefixSpan


class CloSpan(PrefixSpan):
    def __init__(self, ds):
        super(CloSpan, self).__init__(ds)
        self._seen_patterns = []

    def __call__(self, k):
        self._k = k

        # Starting entries : all tid with pid set to -1
        starting_entries = [(i, -1) for i in range(len(self._dataset._db))]
        starting_pattern = []  # Void pattern

        self._main_recursive(starting_pattern, starting_entries)

        # Wait all threads
        for thread in self._threads:
            thread.join()

        # Last filter to keep only closed patterns
        self._results = list(
            starfilter(
                lambda s, pattern, m, p, n: self._is_closed(pattern, p, n),
                self._results,
            )
        )

        # Output results on stdout
        IO.to_stdout(self._dataset, self._results)

        return self._results

    def _contains(self, a, b):
        return tuple(a) in combinations(b, len(a))  # Check if b contains a

    def _is_closed(self, pattern, p, n):
        for result_support, result_pattern, _, result_p, result_n in self._results:
            # If pattern shares all values, is smaller and is contained in the result_pattern :: False
            if (
                p == result_p
                and n == result_n
                and len(result_pattern) > len(pattern)
                and self._contains(pattern, result_pattern)
            ):
                return False
        return True

    def _sum_negative_positive_support(self, matches, p):
        # sum of all remaining lengths
        ns = sum(
            [len(self._dataset.transactions[tid][pid:]) for tid, pid in matches[p:]]
        )
        ps = sum(
            [len(self._dataset.transactions[tid][pid:]) for tid, pid in matches[:p]]
        )
        return ns, ps

    def _prunable(self, pattern, ns, ps):
        # Check if the pattern is prunable
        for seen_pattern, p, n in self._seen_patterns:
            if n == ns and p == ps and self._contains(pattern, seen_pattern):
                return True
        return False

    def _update_results(self, pattern, matches, score, p, n, ps, ns):
        from bisect import insort

        self._seen_patterns.append((pattern, ps, ns))

        # If the result is already contained in the results OR \
        #        we already have `k` support values AND the support is lower than the least best support we already have
        if (
            (score, pattern, matches, p, n) in self._results
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
        insort(self._results, (score, pattern, matches, p, n))

    def _main_recursive(self, pattern, matches, support=0, p=0, n=0, ps=0, ns=0):
        # If we have a pattern, update the results list
        if len(pattern) > 0:
            self._update_results(pattern, matches, support, p, n, ps, ns)

        # Get the next entries (tid, pid)
        new_entries = self.next_entries(matches)
        # Add the score and support for each entry
        # new_entries_score_support = list(Pool().map(self.next_entries_worker, new_entries.items()))
        new_entries_score_support = [
            (item, matches, *self._get_score_key(matches, return_supports=True))
            for item, matches in new_entries.items()
        ]
        # Sort the entries by score
        new_entries_score_support.sort(key=lambda x: x[2], reverse=True)

        # For each item and list new_matches, we extend the tree by recursively call the main function on the new patterns
        for new_item, new_matches, score, support, p, n in new_entries_score_support:

            # Already have results for this k and the support is lower than the kth best, prune.
            if self.current_number_of_k == self._k and score < self.least_best_score:
                break

            # Current pattern + new item
            new_pattern = pattern + [new_item]

            # Check if needed to prune the search tree
            ns, ps = self._sum_negative_positive_support(new_matches, p)
            if self._prunable(new_pattern, ns, ps):
                continue

            # Call the main function on the new pattern
            next_thread = Thread(
                target=self._main_recursive,
                args=(new_pattern, new_matches, support, p, n, ps, ns),
                daemon=True,
            )
            self._threads.append(next_thread)
            next_thread.start()
            # self._main_recursive(new_pattern, new_matches, support)
