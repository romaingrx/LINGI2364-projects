#!/bin/bash

MIN_SUPP=10
FOLD=10
K=$(seq 1 1000)
POS="data/molecules-small.pos"
NEG="data/molecules-small.neg"

TOTRUN=$(($FOLD*3*1000))

CSV="benchmark.csv"

echo "file,k,fold,train_acc,test_acc" > $CSV

algo="Decision Tree"

function launch_(){
    N=0
    file=$1
    for l in $(exec $2 | grep -e "accuracy" -e "fold" | awk '{print $NF}'); do
        case "$N" in 
            "0")
                fold=$l;;
            "1")
                train_acc=$l;;
            "2")
                test_acc=$l
    
                echo "$file,$k,$fold,$train_acc,$test_acc" >> $CSV
    
                N=-1
                ;;
        esac
        
        N=$(($N+1))
    done
}

function launch(){
    case $1 in 
        "02_decision_tree.py" | "03_sequential_covering.py")
            launch_ $1 "python $1 $POS $NEG $2 $MIN_SUPP $FOLD -b";;
        "04_another_classifier.py")
            launch_ $1 "python $1 $POS $NEG $FOLD --top_k $2 --min_supp $MIN_SUPP -b";;
    esac
}


for k in $K; do
    launch "02_decision_tree.py" $k &
    launch "03_sequential_covering.py" $k &
    launch "04_another_classifier.py" $k &
done

# tail -f -n $TOTRUN $CSV | tqdm --total $TOTRUN | wc -l
l=0
while [ $l -le $TOTRUN ]; do
    l=$(cat $CSV | wc -l)
    echo -ne "$l/$((TOTRUN+1)) \r"
done

echo ""
echo "DONE"
