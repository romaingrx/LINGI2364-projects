#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 14, 19:06:36
@last modified : 2021 Apr 14, 19:38:34
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

d0_k5 = np.zeros((5, 5), dtype=np.int_)
d0_k5[1, :1] = [3]
d0_k5[2, :2] = [4, 5]
d0_k5[3, :3] = [6, 2, 3]
d0_k5[4, :4] = [7, 6, 4, 8]

d1_k10 = np.zeros((5, 5), dtype=np.int_)
d1_k10[1, :1] = [0]
d1_k10[2, :2] = [0, 7]
d1_k10[3, :3] = [1, 4, 5]
d1_k10[4, :4] = [1, 5, 6, 8]

d2_k20 = np.zeros((5, 5), dtype=np.int_)
d2_k20[1, :1] = [5]
d2_k20[2, :2] = [5, 12]
d2_k20[3, :3] = [6, 13, 15]
d2_k20[4, :4] = [4, 10, 9, 8]


def plot_ltri(mat: np.ndarray, title: str, cbar: bool = False):
    col_names = ["A", "B", "C", "D", "E"]
    mask = np.triu(np.ones(mat.shape)).astype(np.bool_)
    plt.figure()
    sns.heatmap(
        mat,
        mask=mask,
        xticklabels=col_names,
        yticklabels=col_names,
        cmap="BuPu",
        annot=True,
        cbar=cbar,
    ).set_title(title)
    plt.savefig("./results/plots/" + title.replace(" ", "_"), transparent=True)


plot_ltri(d0_k5, "Dataset Test (0), with k=5")
plot_ltri(d1_k10, "Dataset Protein (1), with k=10")
plot_ltri(d2_k20, "Dataset Reuters (2), with k=20")
