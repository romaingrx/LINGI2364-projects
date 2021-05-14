LINGI2364 projects
===


Project 1 : frequent itemset miner
--- 

```
usage: frequent_itemset_miner.py [-h] -f FILENAME -m MINFREQUENCY -a {apriori,fpgrowth} [-c]

optional arguments:
  -h, --help            show this help message and exit
  -c, --csv             If we want to output as csv format

required arguments:
  -f FILENAME, --filename FILENAME
                        Path to the filename
  -m MINFREQUENCY, --minfrequency MINFREQUENCY
                        Minimum frequency
  -a {apriori,fpgrowth}, --algo {apriori,fpgrowth}
                        Algorithm
```

```
usage: benchmark.sh -f FILENAME -o OUTPUT [--force]

optional arguments:
  --force             Write to the OUTPUT csv file even if it exists

required arguments:
  -f FILENAME, --filename FILENAME
                        Path to the filename dataset
  -o OUTPUT, --output OUTPUT
                        The output csv file
```

Project 2 : Implementing Sequence Mining
---

```
usage: {0*.py}.py [-h] [-c CPROFILE]
                              positive_filepath negative_filepath k

positional arguments:
  positive_filepath     Path to the positive file
  negative_filepath     Path to the negative file
  k                     The number of top sequential patterns

optional arguments:
  -h, --help            show this help message and exit
  -c CPROFILE, --cprofile CPROFILE
                        Run the cprofiler
```

Project 3 : Classifying Graphs
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
