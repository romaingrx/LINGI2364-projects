#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mai 02, 15:58:06
@last modified : 2021 Mai 02, 19:04:45
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
from copy import deepcopy
from bisect import insort
from sklearn import metrics
from sklearn import naive_bayes
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier

from gspan_mining import gSpan
from gspan_mining import GraphDatabase


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = (
            database
        )  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print(
            "Please implement the store function in a subclass for a specific mining task!"
        )

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print(
            "Please implement the prune function in a subclass for a specific mining task!"
        )


class FrequentPositiveGraphs(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive graphs.
    This class provides a method to build a feature matrix for each subset.
    """

    def __init__(self, minsup, database, subsets, k):
        """
        Initialize the task.
        :param minsup: the minimum positive support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        self.patterns = (
            list()
        )  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets

        self._k = k
        self._score_counts = defaultdict(int)

    @property
    def least_best_score(self):
        """least_best_support.
        :return: the value of the least best support
        """
        return self.patterns[0][0] if self.patterns else (-float("inf"), -float("inf"))

    def about_least_best_score(self, c, f):
        """about_least_best_score.

        :param c: the confidence of the new pattern
        :param f: the frequency of the new pattern
        :return:
            -1 if the new pattern is not improving our patterns
             0 if the new pattern has the same confidence/frequency of the least best pattern
             1 if the new pattern is better than the least best pattern
        """
        score = (c, f)

        if score == self.least_best_score or (c, f) in self._score_counts:
            return 0

        return (-1) ** (score < self.least_best_score)

    @property
    def current_number_of_k(self):
        """current_number_of_k.
        :return: the number of current supports
        """
        return len(self._score_counts)

    def delete_least_best(self):
        """delete_least_best.
        delete the least best score in our patterns
        """
        least_best_score = self.least_best_score
        del self._score_counts[least_best_score]

        self
        self.patterns = list(
            filter(lambda result: result[0] != least_best_score, self.patterns)
        )

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        positive = len(gid_subsets[0])
        negative = len(gid_subsets[2])

        frequency = positive + negative

        pos_confidence = positive / frequency
        neg_confidence = negative / frequency

        confidence = max(pos_confidence, neg_confidence)
        label = 1 if pos_confidence >= neg_confidence else -1

        about = self.about_least_best_score(confidence, frequency)

        # If we already have enough top k and the new one is not improving our patterns
        if about < 0 and self.current_number_of_k >= self._k:
            return

        # If the new one is better than least best and we are full of top_k patterns, delete the least best one
        if about > 0 and self.current_number_of_k == self._k:
            self.delete_least_best()

        # Add the new pattern
        insort(self.patterns, [(confidence, frequency), dfs_code, gid_subsets, label])
        self._score_counts[(confidence, frequency)] += 1

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        positive = len(gid_subsets[0])
        # second subset is the set of negative ids
        negative = len(gid_subsets[2])
        # Get the global frequency over positive and negative supports
        frequency = positive + negative
        return frequency < self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for _, _, gid_subsets, _ in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]


def train_evaluate_sequential_covering():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser("Find subgraphs")
    parser.add_argument("positive_file", type=str)
    parser.add_argument("negative_file", type=str)
    parser.add_argument("top_k", type=int)
    parser.add_argument("min_supp", type=int)
    parser.add_argument("n_folds", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.positive_file):
        print("{} does not exist.".format(args.positive_file))
        sys.exit()
    if not os.path.exists(args.negative_file):
        print("{} does not exist.".format(args.negative_file))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        args.positive_file
    )  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        args.negative_file
    )  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if args.n_folds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids,  # Negative test set
        ]
        # Printing fold number:
        print("fold {}".format(1))
        train_and_evaluate(args.min_supp, graph_database, subsets, args.top_k)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // args.n_folds
        neg_fold_size = len(neg_ids) // args.n_folds
        for i in range(args.n_folds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate(
                    (pos_ids[: i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size :])
                ),  # Positive training set
                pos_ids[
                    i * pos_fold_size : (i + 1) * pos_fold_size
                ],  # Positive test set
                numpy.concatenate(
                    (neg_ids[: i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size :])
                ),  # Negative training set
                neg_ids[
                    i * neg_fold_size : (i + 1) * neg_fold_size
                ],  # Negative test set
            ]
            # Printing fold number:
            print("fold {}".format(i + 1))
            train_and_evaluate(args.min_supp, graph_database, subsets, args.top_k)


def remove(list1, list2):
    for elem in range(len(list1)):
        list_1, list_2 = list1[elem], list2[elem]
        for item in list_1:
            list_2.remove(item)
    return list2


def train_and_evaluate(minsup, database, subsets, top_k):
    y_test = [(item, 1) for item in subsets[1]] + [(item, -1) for item in subsets[3]]
    y_test.sort()

    sc_subsets = [
        subset.tolist() if type(subset) != list else subset.copy() for subset in subsets
    ]

    rules = list()
    y_test_predicted = list()

    for k in range(top_k):
        task = FrequentPositiveGraphs(minsup, database, sc_subsets, 1)
        gSpan(task).run()

        if task.patterns:

            task.patterns.sort(key=lambda x: (x[1], *x[0], x[2]))
            best_pattern = task.patterns[0]
            (confidence, frequency), dfs_code, gid_subsets, label = best_pattern

            rules.append(best_pattern)

            for item in gid_subsets[1] + gid_subsets[3]:
                insort(y_test_predicted, (item, label))

            sc_subsets = remove(gid_subsets, sc_subsets)

    default_label = 1 if len(sc_subsets[0]) >= len(sc_subsets[2]) else -1

    for item in sc_subsets[1] + sc_subsets[3]:
        insort(y_test_predicted, (item, default_label))

    for (confidence, frequency), dfs_code, _, _ in rules:
        print(f"{dfs_code} {confidence} {frequency}")

    predicted_labels = [label for _, label in y_test_predicted]
    print(predicted_labels)

    accuracy = sum(t == p for t, p in zip(y_test, y_test_predicted)) / len(
        y_test_predicted
    )
    print(f"accuracy: {accuracy}")

    print()


if __name__ == "__main__":
    train_evaluate_sequential_covering()
