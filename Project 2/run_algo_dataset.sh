#!/bin/bash

reset='\e[0m'
red='\e[0;91m'
purple='\e[0;95m'

ROOT="$(pwd)/Datasets"
POSITIVE=""
NEGATIVE=""

function load_files() {
    ROOT_DATASET="$ROOT/$1"
    POSITIVE="$ROOT_DATASET/$2"
    NEGATIVE="$ROOT_DATASET/$3"
    # echo -e "${red}LOADING ::${reset} $1"
    # echo "    positive :: $POSITIVE"
    # echo "    negative :: $NEGATIVE"
}

case $2 in
    0)
        load_files "Test" "positive.txt" "negative.txt";;
    1)
        load_files "Protein" "SRC1521.txt" "PKA_group15.txt";;
    2)
        load_files "Reuters" "earn.txt" "acq.txt";;
esac

python $1 "${POSITIVE}" "${NEGATIVE}" "$3" -c "$4"
