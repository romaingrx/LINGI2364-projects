
Project 3 : Classifying Graphs
===

The aim of this project is to train several models to predict labels on patterns in a supervised dataset with sequences.


How to use scripts?
---


```
usage: 0[1-3].*.py [-h] [-b]
                      positive_file negative_file top_k min_supp n_folds

positional arguments:
  positive_file
  negative_file
  top_k
  min_supp
  n_folds

optional arguments:
  -h, --help       show this help message and exit
  -b, --benchmark
```

```
usage: 04_another_classifier.py [-h] [-k TOP_K] [-s MIN_SUPP] [-b]
                      positive_file negative_file n_folds

positional arguments:
  positive_file
  negative_file
  n_folds

optional arguments:
  -h, --help            show this help message and exit
  -k TOP_K, --top_k TOP_K
  -s MIN_SUPP, --min_supp MIN_SUPP
  -b, --benchmark
```
