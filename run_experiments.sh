#!/bin/bash

# A script to run train.py sequentially with different --num_points values.
# It saves logs and checkpoints in separate directories per run to avoid overwriting.

# Usage: ./run_experiments.sh [num_points1] [num_points2] ...
# Example: ./run_experiments.sh 256 512 1024 2048

# Default list of num_points if no arguments are provided
if [ $# -eq 0 ]; then
    POINTS_LIST=(256 512 1024)
    echo "No num_points arguments provided. Using default list: ${POINTS_LIST[@]}"
else
    POINTS_LIST=("$@")
fi

BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

for pts in "${POINTS_LIST[@]}"; do
    echo "=========================================================="
    echo " Starting training for num_points = $pts"
    echo "=========================================================="
    
    # Create a specific directory for this run's checkpoints and logs
    RUN_DIR="${BASE_DIR}/num_points_${pts}"
    mkdir -p "$RUN_DIR"
    
    LOG_FILE="${RUN_DIR}/training.log"
    
    # Run the python script with unbuffered output so logs are written in real-time.
    # Both stdout and stderr are redirected to the log file and printed to console using tee.
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python train.py \
        --num_points "$pts" \
        --save_dir "$RUN_DIR" \
        2>&1 | tee "$LOG_FILE"
        
    echo "=========================================================="
    echo " Finished training for num_points = $pts"
    echo " Checkpoints and logs saved in: $RUN_DIR"
    echo "=========================================================="
    echo ""
done

echo "All training runs completed!"
