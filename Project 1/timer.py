#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 12, 17:12:39
@last modified : 2021 Mar 12, 17:21:34
"""

from frequent_itemset_miner import apriori, alternative_miner
import argparse
from time import perf_counter


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='Path to the filename', type=str, required=True)
parser.add_argument('-m','--minfrequency', help='Minimum frequency', type=float, required=True)
parser.add_argument('-a','--algo', help='Algorithm', choices=['apriori', 'fpgrowth'], type=str, required=True)

args = parser.parse_args()

f = apriori if args.algo == 'apriori' else alternative_miner

tic = perf_counter()
f(args.filename, args.minfrequency)
tac = perf_counter()

csvvalues = [args.algo, args.filename, str(args.minfrequency), f"{tac-tic}"]
print(",".join(csvvalues))
