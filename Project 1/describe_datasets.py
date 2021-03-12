#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 12, 22:04:43
@last modified : 2021 Mar 12, 22:53:09
"""

import os
import numpy as np
import pandas as pd
from frequent_itemset_miner import Dataset

def get_data(ds):
    lengths = list(map(len, ds._transactions))
    return dict(
            mean_items_per_trans = np.mean(lengths),
            std_items_per_trans = np.std(lengths),
            min_items_per_trans = np.min(lengths),
            max_items_per_trans = np.max(lengths),
            n_items_total = len(ds._items),
            n_transactions_total = len(ds._transactions),
            )

def get_all():
    all_data = dict()
    datasets = os.path.join(os.curdir, "Datasets")
    for f in os.listdir(datasets):
        if f.split('.')[-1] == "dat":
            fname = os.path.join(datasets, f)
            ds = Dataset(fname)
            all_data[f] = get_data(ds)
    pd.DataFrame(all_data).to_csv("all_data.csv")



get_all()
