#!/bin/bash

export NMODELS=512
export NNODES=${1:-20} # use `20` if the first argument is not given.
export ROOT=$NMODELS-model-$NNODES-node-exp


# Initial baseline.
python experiments/input_search.py --max_nodes $NNODES --n_model $NMODELS --max_sample 1 --max_time_ms 8 --root $ROOT --result result-0.csv

# 7 data points
for i in {1..7}
do
    echo "Running experiment $i"
    python experiments/input_search.py --max_nodes $NNODES --n_model $NMODELS --max_sample 1024 --max_time_ms $(($i * 8)) --root $ROOT --result result-$i.csv
done
