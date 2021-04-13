#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 07, 13:03:42
@last modified : 2021 Mar 12, 18:55:44
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

__authors__ = "<24, Romain Graux, Francois Gouverneur>"
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

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for trans_num, line in enumerate(lines):
                transaction = list(map(int, line.split(" ")))
                self._transactions.append(transaction)
                for item in transaction:
                    self._items.add(item)

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
    """Apriori."""

    @staticmethod
    def cover(ds, itemset):
        """cover.
        return the cover of the itemset in the dataset `ds`

        :param ds: an instance of Dataset
        :param itemset: a tuple representing an itemset (ex: (a,b,c))
        """
        cov = list(filter(set(itemset).issubset, ds._transactions))
        #cov = list()
        #for t, itemsetD in enumerate(ds._transactions):
        #    if itemset.issubset(set(itemsetD)):
        #        cov.append(t)
        return cov

    @staticmethod
    def support(ds, itemset):
        """support.
        return the support of the itemset in the dataset `ds`

        :param ds: an instance of Dataset
        :param itemset: a tuple representing an itemset (ex: (a,b,c))
        """
        cov = Apriori.cover(ds, itemset)
        return len(cov)

    @staticmethod
    def generate_candidates(previous_candidates):
        """generate_candidates.
        generate the next candidates (next level) from the previous candidates

        :param previous_candidates: a list containing the previous candidates
        """
        candidates = []
        for i, a in enumerate(previous_candidates):
            for b in previous_candidates[i:]:
                if a[:-1] == b[:-1] and a != b:
                    candidate = list(a) + [b[-1]]
                    candidates.append(tuple(sorted(candidate)))
        return candidates


    @staticmethod
    def apriori(filepath, minFrequency:float):
        """apriori.
        return the itemsets respecting the minimum frequency `minFrequency`

        :param filepath: the path to the dataset
        :param minFrequency: the minimum frequency for which we want the itemsets
        """
        assert 0 <= minFrequency <= 1, f"the minimum frequency has to be between 0 and 1, :: {minFrequency}"
        ds = Dataset(filepath)
        
        items = list(chain(*ds._transactions)) # list items -> mettre transactions à plats
        items_counter = dict(Counter(items))   # compte chaque item
        tracker = {(item,):support/ds.trans_num() for item, support in items_counter.items() if support/ds.trans_num() >= minFrequency} # on a tt les items qui respectent la fréquence
        candidates = list(tracker.keys()) 
        
        while True:
            itemsets = Apriori.generate_candidates(candidates) # generate candidates énumère tt les candidats (check juste si la fin est diff)
            if len(itemsets) == 0:
                break

            frequencies = list(map(lambda i:Apriori.support(ds, i)/ds.trans_num(), itemsets))
            valids = {s:freq for s, freq in zip(itemsets, frequencies) if freq >= minFrequency}
            candidates = list(valids.keys())

            tracker.update(valids) # rajoute les nouveaux

        return tracker

class Node:
    """Node."""

    def __init__(self, name, frequency, parent):
        """__init__.

        :param name: the item name
        :param frequency: the frequency of the item in this tree
        :param parent: the parent of the node 
        """
        self.name = name
        self._frequency = frequency
        self.parent = parent
        self.children = dict()
        self.next = None

    def inc(self, amount):
        """inc.
        increment the frequency by an amount

        :param amount: the amount to increment
        """
        self._frequency += amount

    def __str__(self, lvl=0):
        ind = "| " * lvl
        s = f"{ind}{self.name}->{self._frequency}\n"
        for node in self.children.values():
            s += node.__str__(lvl=lvl + 1)
        return s


class FPgrowth:
    """FPgrowth."""

    @staticmethod
    def updateTable(item, node, table):
        """updateTable.
        update the table with the node and item as leaf

        :param item: the item in the table
        :param node: the node to be put as leaf
        :param table: the current table
        """
        if table[item][1] is None:
            table[item][1] = node
        else:
            runner = table[item][1]
            while runner.next is not None:
                runner = runner.next
            runner.next = node

    @staticmethod
    def updateTree(item, node, table, frequency):
        """updateTree.
        update the tree with frequency for node and item

        :param item: the item in the children tree
        :param node: the current node
        :param table: the table to update
        :param frequency: the amount of frequency for item
        """
        if item in node.children:
            node.children[item].inc(frequency)
        else:
            newNode = Node(item, frequency, node)
            node.children[item] = newNode
            FPgrowth.updateTable(item, newNode, table)
        return node.children[item]

    @staticmethod
    def constructTree(itemsets, frequencies, minSupport):
        """constructTree.
        construct the tree based on the itemsets with frequencies respecting the minSupport

        :param itemsets: a list of itemset
        :param frequencies: the corresponding frequencies
        :param minSupport: the desired minimum support
        """
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
        """prefixPath.
        return the prefix path

        :param item: the item from which we want the path
        :param table: the current table
        """
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
        """getSupport.
        return the support of the prefix in the itemsets

        :param prefix: a prefix for which we want the support
        :param itemsets: a list of itemsets
        """
        return sum(list(map(set(prefix).issubset, itemsets)))

    @staticmethod
    def fpgrowth(filename: str, minFrequency):
        """fpgrowth.
        application of the whole fpgrowth algorithm

        :param filename: the path to the dataset
        :param minFrequency: the desired minimum frequency
        """
        def miner(table, minSupport, prefix, prefixTracker):
            """miner.
            mine from prefix in the table

            :param table: the current table
            :param minSupport: the desired minimum support
            :param prefix: the current prefix
            :param prefixTracker: a list for tracking the prefixes
            """
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
    """to_stdout.
    print the good itemsets on the stdout with the desired format

    :param support_itemset: the good itemsets to print
    """
    for itemset, freq in support_itemset.items():
        l = list(itemset)
        l.sort()
     #   print(str(l) + "(" + str(freq) + ")")


def apriori(filepath, minFrequency, stdout=True):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    support_itemset = Apriori.apriori(filepath, minFrequency)
    if stdout:
        to_stdout(support_itemset)
    return support_itemset


def alternative_miner(filepath, minFrequency, stdout=True):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    support_itemset = FPgrowth.fpgrowth(filepath, minFrequency)
    if stdout:
        to_stdout(support_itemset)
    return support_itemset

if __name__ == "__main__":
    import os

    datasets = os.path.join(os.curdir, "Datasets")
   # print(len(datasets._items)) 
    
    fname = os.path.join(datasets, "toy.dat")
    db = Dataset(fname)
    
    print("TOYS\n")
    
    #for freq in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]:
    #for freq in [0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.86, 0.84, 0.82, 0.8]:#, 0.45, 0.4, 0.35, 0.3, 0.25]:
   # for freq in [0.7, 0.675, 0.65, 0.625, 0.6, 0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3]:
    for freq in [0.95]:
        
        tic = time()
        itemsets = apriori(fname, freq)
        toc = time()
        print("Time Apriori : " + str(toc-tic) + ", freq : " + str(freq))
        list1 = toc-tic
        
        tic = time()
        itemsets = alternative_miner(fname, freq)
        toc = time()
        print("Time Alternative : " + str(toc-tic) + ", freq : " + str(freq))
        list2 = toc-tic
        
        print()
        print() 
    
print(list1)
print(list2)
