
Project 1 : frequent itemset miner
===

frequent_itemset_miner.py
---

```
usage: frequent_itemset_miner.py [-h] -f FILENAME -m MINFREQUENCY -a {apriori,fpgrowth} [-c]

optional arguments:
  -h, --help            show this help message and exit
  -c, --csv             If we want to output as csv format

required arguments:
  -f FILENAME, --filename FILENAME
                        Path to the filename dataset
  -m MINFREQUENCY, --minfrequency MINFREQUENCY
                        Minimum frequency
  -a {apriori,fpgrowth}, --algo {apriori,fpgrowth}
                        Algorithm
```

benchmark.sh
---

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
