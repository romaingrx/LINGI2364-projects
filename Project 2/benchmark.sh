#!/bin/bash

OUTPUT_CSV="benchmark.csv"
HEADER_CSV="pyfile,dataset,k,number_top_k,time"
PY_FILES=$(ls | grep "0[1-9]1*.*.py")
N_PYFILES=$(ls | grep "0[1-9]1*.*.py" | wc -l)

if [ $# -eq 1 ]; then 
    K=$1
else
    K=30
fi

N_TOTAL=$(($N_PYFILES*$K*3))

# First output the header of the csv
echo ${HEADER_CSV} > ${OUTPUT_CSV}

# Log 
echo -ne "          "
echo "${HEADER_CSV}" | column -t -s ","

i=1
for pyfile in ${PY_FILES}; do
    for dataset in {0..2}; do 
        for k in $(seq 1 $K); do

            tic=$(date +%s.%N)
            # Count the number of top k
            ntopk=$(./run_algo_dataset.sh ${pyfile} ${dataset} ${k} 0 | wc -l)
            tac=$(date +%s.%N)

            # Compute the taken time
            taken_time=$(echo "$tac - $tic" | bc)

            # Output the line in the csv file
            csv_line="$pyfile,$dataset,$k,$ntopk,$taken_time"
            echo $csv_line >> ${OUTPUT_CSV}

            # Log
            echo -ne "$i/$N_TOTAL :: $(echo $csv_line | column -t -s ",")\r"
            i=$(($i+1))
        done
    done
done
