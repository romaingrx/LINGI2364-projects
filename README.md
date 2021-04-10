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

Project 2 : Implementing Sequence Mining
---

```
usage: {ANY}.py [-h] positive_filepath negative_filepath k

positional arguments:
  positive_filepath  Path to the positive file
  negative_filepath  Path to the negative file
  k                  The number of top sequential patterns

optional arguments:
  -h, --help         show this help message and exit
```
