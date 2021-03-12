#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 07, 13:03:42
@last modified : 2021 Mar 12, 09:29:36
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
from itertools import combinations
from collections import defaultdict
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
        cov = list()
        for t, itemsetD in enumerate(ds._transactions):
            if itemset.issubset(set(itemsetD)):
                cov.append(t)
        return cov
    
    @staticmethod
    def support(ds, itemset):
        cov = Apriori.cover(ds, itemset)
        return len(cov)
    
    @staticmethod
    def create_next_gen(Ck, k):
        next_gen = []
        for idx, ca in enumerate(Ck):
            for cb in Ck[idx+1:]:
                la = list(ca)[:k-2]; lb = list(cb)[:k-2]
                la.sort(); lb.sort()
                if la == lb:
                    next_gen.append(ca|cb)
        return np.array(next_gen)
    
    @staticmethod
    def remove_non_supported(ds, Ck, minFrequency):
        freqs = np.array([Apriori.support(ds, c)/ds.trans_num() for c in Ck])
        valid = freqs >= minFrequency
        return Ck[valid], freqs[valid]
    
    @staticmethod
    def createC1(ds):
        C1 = list(ds._items)
        C1.sort()
        return np.array(list(map(frozenset, [[c] for c in C1])))
    
    
    
    @staticmethod
    def apriori(filepath, minFrequency):
        """Runs the apriori algorithm on the specified file with the given minimum frequency"""
        ds = Dataset(filepath)
        D = list(map(set, ds._transactions))
        support_itemset = dict()
    
        C1 = Apriori.createC1(ds)
        L, f = Apriori.remove_non_supported(ds, C1, minFrequency)
    
        support_itemset.update(dict(zip(L, f)))
        k = 2
    
        while len(L) > 0:
            C = Apriori.create_next_gen(L, k)
            L, f = Apriori.remove_non_supported(ds, C, minFrequency)
            support_itemset.update(dict(zip(L, f)))
            k += 1

        return support_itemset



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

    def __call__(self, dataset:Dataset, minFrequency:float):
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
                condTree, newTable = FPgrowth.constructTree(condPaths, frequencies, minSupport)
                miner(newTable, minSupport, newPrefix, prefixTracker)

        dataset = Dataset(filename)
    
        prefixTracker = []
        itemsets = dataset._transactions
        minSupport = minFrequency * len(itemsets)
        tree, table = FPgrowth.constructTree(itemsets, [1] * len(itemsets), minSupport)
        miner(table, minSupport, set(), prefixTracker)
    
        support_itemset = dict()
    
        for prefix in prefixTracker:
            support_itemset[frozenset(prefix)] = FPgrowth.getSupport(prefix, itemsets) / len(itemsets)
    
        return support_itemset

def to_stdout(support_itemset):
    for itemset, freq in support_itemset.items():
        l = list(itemset)
        l.sort()
        print(
                f"[{','.join([str(i) for i in l])}]({freq})"
                )


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

if __name__ == '__main__':
    import os
    datasets = os.path.join(os.curdir, "Datasets")
    fname = os.path.join(datasets, "toy.dat")

    db = Dataset(fname)
    itemsets = apriori(fname, .5)
    print()
    itemsets = alternative_miner(fname, .5)
