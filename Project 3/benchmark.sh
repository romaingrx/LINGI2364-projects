#!/bin/bash

MIN_SUPP=5
FOLD=4
MAX_K=2
POS="data/molecules-small.pos"
NEG="data/molecules-small.neg"

TOTRUN=$(($FOLD*3*$MAX_K))
K=$(seq 1 $MAX_K)

CMD=".commands"
CSV="benchmark.csv"

echo "file,k,fold,train_acc,test_acc" > $CSV

algo="Decision Tree"

function launch_(){
    N=0
    CSV="$1"
    file=$(echo $2 | awk '{print $2}')
    for l in $(bash -c "$2" | grep -e "accuracy" -e "fold" | awk '{print $NF}'); do
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
            # launch_ $1 "python $1 $POS $NEG $2 $MIN_SUPP $FOLD -b";;
            cmd="python $1 $POS $NEG $2 $MIN_SUPP $FOLD -b";;
        "04_another_classifier.py")
            # launch_ $1 "python $1 $POS $NEG $FOLD --top_k $2 --min_supp $MIN_SUPP -b";;
            cmd="python $1 $POS $NEG $FOLD --top_k $2 --min_supp $MIN_SUPP -b";;
    esac
    echo "$cmd" >> $CMD
}

echo -n "" > CMD

for k in $K; do
    launch "02_decision_tree.py" $k
    launch "03_sequential_covering.py" $k
    launch "04_another_classifier.py" $k
done

export -f launch_
cat $CMD | parallel -j 20 launch_ $CSV &

# tail -f -n $TOTRUN $CSV | tqdm --total $TOTRUN | wc -l
l=0
while [ $l -le $TOTRUN ]; do
    l=$(cat $CSV | wc -l)
    echo -ne "$l/$((TOTRUN+1)) \r"
done

echo ""
echo "DONE"

rm -f $CMD
