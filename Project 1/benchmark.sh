#!/bin/bash

ALGOS=("apriori" "fpgrowth")
FREQUENCIES=$(seq 0 .1 1)
DATABASE="./Datasets/toy.dat"
OUT="results.csv"

SWPPY=".exec.py.swp"
HEADER="algo,dataset,frequency,time"

cat timer.py > $SWPPY

echo $HEADER > $OUT

for algo in ${ALGOS[@]}; do
    echo $algo
    for freq in $FREQUENCIES; do
        python $SWPPY -f $DATABASE -m $freq -a $algo >> $OUT
    done
done
