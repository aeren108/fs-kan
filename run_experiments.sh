#!/bin/bash

# A script to run train.py with specific --num_points and --train-set-size values.
# It saves logs and checkpoints in a separate directory per run to avoid overwriting.

# Usage: ./run_experiments.sh <num_points> <train_set_size>
# Example: ./run_experiments.sh 1024 1000

if [ $# -ne 2 ]; then
    echo "Usage: $0 <num_points> <train_set_size>"
    echo "Example: $0 1024 1000"
    exit 1
fi

NUM_POINTS=$1
TRAIN_SET_SIZE=$2

BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

RUN_DIR="${BASE_DIR}/pts_${NUM_POINTS}_train_${TRAIN_SET_SIZE}"
mkdir -p "$RUN_DIR"

echo "=========================================================="
echo " Starting training for num_points = $NUM_POINTS, train_set_size = $TRAIN_SET_SIZE"
echo "=========================================================="

LOG_FILE="${RUN_DIR}/training.log"

# Run the python script with unbuffered output so logs are written in real-time.
# Both stdout and stderr are redirected to the log file and printed to console using tee.
source .venv/bin/activate
PYTHONUNBUFFERED=1 python train.py \
    --num_points "$NUM_POINTS" \
    --save_dir "$RUN_DIR" \
    --train-set-size "$TRAIN_SET_SIZE" \
    --lr 0.01 \
    2>&1 | tee "$LOG_FILE"
    
echo "=========================================================="
echo " Finished training for num_points = $NUM_POINTS, train_set_size = $TRAIN_SET_SIZE"
echo " Checkpoints and logs saved in: $RUN_DIR"
echo "=========================================================="
echo ""
echo "Training run completed!"
