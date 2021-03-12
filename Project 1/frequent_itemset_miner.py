#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 07, 13:03:42
@last modified : 2021 Mar 12, 11:26:32
"""

"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<XXX, Romain Graux, Francois Gouverneur>"
"""

import numpy as np
from itertools import combinations, chain
from collections import defaultdict, Counter
from time import perf_counter as time


class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self._transactions = list()
        self._items = set()
        #        self._tidlists = defaultdict(list)

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for trans_num, line in enumerate(lines):
                transaction = list(map(int, line.split(" ")))
                self._transactions.append(transaction)
                for item in transaction:
                    self._items.add(item)
        #                   self._tidlists[item].append(trans_num)

        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self._transactions)

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self._items)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self._transactions[i]

    def __str__(self):
        return "\n".join([str(t) for t in self._transactions])


class Apriori:
    @staticmethod
    def cover(ds, itemset):
        cov = list(filter(set(itemset).issubset, ds._transactions))
        #cov = list()
        #for t, itemsetD in enumerate(ds._transactions):
        #    if itemset.issubset(set(itemsetD)):
        #        cov.append(t)
        return cov

    @staticmethod
    def support(ds, itemset):
        cov = Apriori.cover(ds, itemset)
        return len(cov)

    @staticmethod
    def gen_cand_2(cand_prev):
        cands = []
        for i, cand_1 in enumerate(cand_prev):
            for cand_2 in cand_prev[i:]:
                cand_1 = list(cand_1)
                cand_2 = list(cand_2)
                if cand_1[:-1] == cand_2[:-1] and cand_1 != cand_2:
                    cand_add = list(cand_1) + [cand_2[-1]]
                    cands.append(tuple(sorted(cand_add)))
        return cands


    @staticmethod
    def apriori(filepath, minFrequency):
        """Runs the apriori algorithm on the specified file with the given minimum frequency"""
        ds = Dataset(filepath)
        
        items = list(chain(*ds._transactions))
        items_counter = dict(Counter(items))
        tracker = {(item,):support/ds.trans_num() for item, support in items_counter.items() if support/ds.trans_num() >= minFrequency}
        candidates = list(tracker.keys()) 
        
        while True:
            itemsets = Apriori.gen_cand_2(candidates)
            if len(itemsets) == 0:
                break

            frequencies = list(map(lambda i:Apriori.support(ds, i)/ds.trans_num(), itemsets))
            valids = {s:freq for s, freq in zip(itemsets, frequencies) if freq >= minFrequency}
            candidates = list(valids.keys())

            tracker.update(valids)

        tracker = {frozenset(itemset):freq for itemset, freq in tracker.items()}
        return tracker


class Node:
    def __init__(self, name, frequency, parent):
        self.name = name
        self._frequency = frequency
        self.parent = parent
        self.children = dict()
        self.next = None

    def inc(self, amount):
        self._frequency += amount

    def __str__(self, lvl=0):
        ind = "| " * lvl
        s = f"{ind}{self.name}->{self._frequency}\n"
        for node in self.children.values():
            s += node.__str__(lvl=lvl + 1)
        return s


class FPgrowth:
    @staticmethod
    def updateTable(item, node, table):
        if table[item][1] is None:
            table[item][1] = node
        else:
            runner = table[item][1]
            while runner.next is not None:
                runner = runner.next
            runner.next = node

    @staticmethod
    def updateTree(item, node, table, frequency):
        if item in node.children:
            node.children[item].inc(frequency)
        else:
            newNode = Node(item, frequency, node)
            node.children[item] = newNode
            FPgrowth.updateTable(item, newNode, table)
        return node.children[item]

    @staticmethod
    def constructTree(itemsets, frequencies, minSupport):
        # Construct the frequency table per item
        table = defaultdict(lambda: 0)
        for frequency, itemset in zip(frequencies, itemsets):
            for item in itemset:
                table[item] += frequency

        # Drop items below `minSupport`
        table = dict(
            (item, support) for item, support in table.items() if support >= minSupport
        )
        # Return if nothing is above minSupport
        if len(table) == 0:
            return None, None

        table = dict((item, [frequency, None]) for item, frequency in table.items())

        # Construct the tree begining with Null root node
        tree = Node("root", 1, None)
        for frequency, itemset in zip(frequencies, itemsets):
            itemset = list(
                filter(lambda x: x in table, itemset)
            )  # Keep only items which are in the table
            itemset.sort(
                key=lambda item: table[item][0], reverse=True
            )  # Sort by the inverse of frequency
            node = tree
            for item in itemset:
                node = FPgrowth.updateTree(item, node, table, frequency)

        return tree, table

    @staticmethod
    def prefixPath(item, table):
        node = table[item][1]
        condPaths = []
        frequencies = []

        while node is not None:
            prefixPath = []

            runner = node
            while runner.parent is not None:
                prefixPath.append(runner.name)
                runner = runner.parent

            if len(prefixPath) > 1:
                condPaths.append(prefixPath[1:])
                frequencies.append(node._frequency)

            node = node.next
        return condPaths, frequencies

    @staticmethod
    def getSupport(prefix, itemsets):
        count = 0
        for itemSet in itemsets:
            if set(prefix).issubset(itemSet):
                count += 1
        return count

    def __call__(self, dataset: Dataset, minFrequency: float):
        return FPgrowth.fpgrowth(dataset, minFrequency)

    @staticmethod
    def fpgrowth(filename: str, minFrequency):
        def miner(table, minSupport, prefix, prefixTracker):
            if table is None:
                return
            sorted_per_freq = sorted(
                list(table.items()), key=lambda l: l[1][0]
            )  # Sort by decreasing frequencies
            items = list(zip(*sorted_per_freq))[0]

            for item in items:
                newPrefix = prefix.copy()
                newPrefix.add(item)
                prefixTracker.append(newPrefix)
                condPaths, frequencies = FPgrowth.prefixPath(item, table)
                condTree, newTable = FPgrowth.constructTree(
                    condPaths, frequencies, minSupport
                )
                miner(newTable, minSupport, newPrefix, prefixTracker)

        dataset = Dataset(filename)

        prefixTracker = []
        itemsets = dataset._transactions
        minSupport = minFrequency * len(itemsets)
        tree, table = FPgrowth.constructTree(itemsets, [1] * len(itemsets), minSupport)
        miner(table, minSupport, set(), prefixTracker)

        support_itemset = dict()

        for prefix in prefixTracker:
            support_itemset[frozenset(prefix)] = FPgrowth.getSupport(
                prefix, itemsets
            ) / len(itemsets)

        return support_itemset


def to_stdout(support_itemset):
    for itemset, freq in support_itemset.items():
        l = list(itemset)
        l.sort()
        print(f"[{','.join([str(i) for i in l])}]({freq})")


def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    support_itemset = Apriori.apriori(filepath, minFrequency)
    to_stdout(support_itemset)
    return support_itemset


def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    # TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
    support_itemset = FPgrowth.fpgrowth(filepath, minFrequency)
    to_stdout(support_itemset)
    return support_itemset


if __name__ == "__main__":
    import os

    datasets = os.path.join(os.curdir, "Datasets")
    fname = os.path.join(datasets, "accidents.dat")
    db = Dataset(fname)
    
    for freq in [0.8,0.85,0.9,0.95]:    
        
        tic = time()
       # itemsets = apriori(fname, freq)
        toc = time()
        print("Time Apriori : " + str(toc-tic) + ", freq : " + str(freq))
        
     #   print()
        tic = time()
        itemsets = alternative_miner(fname, freq)
        toc = time()
        print("Time Alternative : " + str(toc-tic) + ", freq : " + str(freq))
        
        print()
        print()
        
