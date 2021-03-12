#!/bin/bash

Black=$'\e[1;30m'
Red=$'\e[1;31m'

function error(){
    echo "$Red ERROR :: $Black $1"
    exit
}

ALGOS=("apriori" "fpgrowth")
FREQUENCIES=$(seq 0 .1 1)
DATABASE="./Datasets/toy.dat"
OUT="results.csv"

SWPPY=".exec.py.swp"
HEADER="algo,dataset,frequency,time"

FORCE=0


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--output) OUT="$2"; shift ;;
        -f|--filename) DATABASE="$2"; shift ;;
        --force) FORCE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -f $OUT ]] && [[ $FORCE -eq 0 ]]; then
    error "$OUT already exists, please remove it before running benchmark"
fi

cat timer.py > $SWPPY

echo $HEADER > $OUT

echo "run $DATABASE with output $OUT"

for algo in ${ALGOS[@]}; do
    echo $algo
    for freq in $FREQUENCIES; do
        python $SWPPY -f $DATABASE -m $freq -a $algo >> $OUT
    done
done

if [[ -f $SWPPY ]]; then 
    rm $SWPPY
fi
