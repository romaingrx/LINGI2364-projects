#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 09, 17:10:08
@last modified : 2021 Mar 09, 19:48:00
"""

import numpy as np
from itertools import combinations
from collections import defaultdict


class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self._transactions = list()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for trans_num, line in enumerate(lines):
                transaction = list(map(int, line.split(" ")))
                self._transactions.append(transaction)

        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self._transactions)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self._transactions[i]

    def __str__(self):
        return "\n".join([str(t) for t in self._transactions])

    def __repr__(self):
        return str(self)


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


def updateTable(item, node, table):
    if table[item][1] is None:
        table[item][1] = node
    else:
        runner = table[item][1]
        while runner.next is not None:
            runner = runner.next
        runner.next = node


def updateTree(item, node, table, frequency):
    if item in node.children:
        node.children[item].inc(frequency)
    else:
        newNode = Node(item, frequency, node)
        node.children[item] = newNode
        updateTable(item, newNode, table)
    return node.children[item]


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
            node = updateTree(item, node, table, frequency)

    return tree, table


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


def retrieveFreq(prefix, root):
    support = 0
    # for item, child in root.children.items():
    #    if item in prefix:
    #        pass
    return support


def getSupport(prefix, itemsets):
    count = 0
    for itemSet in itemsets:
        if set(prefix).issubset(itemSet):
            count += 1
    return count


def fpgrowth(dataset: Dataset, minFrequency):
    def miner(table, minSupport, prefix, prefixTracker):
        sorted_per_freq = sorted(
            list(table.items()), key=lambda l: l[1][0]
        )  # Sort by decreasing frequencies
        items = list(zip(*sorted_per_freq))[0]

        for item in items:
            newPrefix = prefix.copy()
            newPrefix.add(item)
            prefixTracker.append(newPrefix)
            condPaths, frequencies = prefixPath(item, table)
            condTree, newTable = constructTree(condPaths, frequencies, minSupport)
            newFreq = retrieveFreq(newPrefix, table)
            if newTable is not None:
                miner(newTable, minSupport, newPrefix, prefixTracker)

    prefixTracker = []
    itemsets = dataset._transactions
    minSupport = minFrequency * len(itemsets)
    tree, table = constructTree(itemsets, [1] * len(itemsets), minSupport)
    miner(table, minSupport, set(), prefixTracker)

    results = dict()

    for prefix in prefixTracker:
        results[frozenset(prefix)] = getSupport(prefix, itemsets) / len(itemsets)

    for itemset, freq in results.items():
        print(
                f"[{','.join([str(i) for i in list(itemset)])}]({freq})"
                )

    return results


if __name__ == "__main__":
    import os

    datasets = os.path.join(os.curdir, "Datasets")
    fname = os.path.join(datasets, "toy.dat")

    db = Dataset(fname)

    freq = fpgrowth(db, 0.5)
