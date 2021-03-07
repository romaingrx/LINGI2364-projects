#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 07, 13:03:42
@last modified : 2021 Mar 07, 16:38:49
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

class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self._transactions = list()
        self._items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for line in lines:
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

def cover(ds, itemset):
    cov = list()
    for t, itemsetD in enumerate(ds._transactions):
        if itemset.issubset(set(itemsetD)):
            cov.append(t)
    return cov

def support(ds, itemset):
    cov = cover(ds, itemset)
    return len(cov)

def create_next_gen(Ck, k):
    next_gen = []
    for idx, ca in enumerate(Ck):
        for cb in Ck[idx+1:]:
            la = list(ca)[:k-2]; lb = list(cb)[:k-2]
            la.sort(); lb.sort()
            if la == lb:
                next_gen.append(ca|cb)
    return np.array(next_gen)

def remove_non_supported(ds, Ck, minFrequency):
    freqs = np.array([support(ds, c)/ds.trans_num() for c in Ck])
    valid = freqs >= minFrequency
    return Ck[valid], freqs[valid]

def createC1(ds):
    C1 = list(ds._items)
    C1.sort()
    return np.array(list(map(frozenset, [[c] for c in C1])))

def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    ds = Dataset(filepath)
    D = list(map(set, ds._transactions))
    support_itemset = dict()

    C1 = createC1(ds)
    L, f = remove_non_supported(ds, C1, minFrequency)

    support_itemset.update(dict(zip(L, f)))
    k = 2

    while len(L) > 0:
        C = create_next_gen(L, k)
        L, f = remove_non_supported(ds, C, minFrequency)
        support_itemset.update(dict(zip(L, f)))
        k += 1

    for itemset, freq in support_itemset.items():
        print(
                f"[{','.join([str(i) for i in list(itemset)])}]({freq})"
                )

    return support_itemset


def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    # TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
    raise NotImplementedError()

if __name__ == '__main__':
    import os
    datasets = os.path.join(os.curdir, "Datasets")
    fname = os.path.join(datasets, "toy.dat")

    itemsets = apriori(fname, .5)
