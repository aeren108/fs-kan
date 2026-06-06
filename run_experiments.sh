#!/bin/bash

# A script to run train.py over a grid of --num_points × --train-set-size values.
# Supports sequential (default) and parallel execution with configurable concurrency.
#
# Usage:
#   Sequential:  ./run_experiments.sh --points "256 512 1024" --sizes "500 1000 2000"
#   Parallel:    ./run_experiments.sh --points "256 512 1024" --sizes "500 1000" --parallel 3
#
# Options:
#   --points   "P1 P2 ..."   Space-separated list of num_points values (required)
#   --sizes    "S1 S2 ..."   Space-separated list of train_set_size values (required)
#   --parallel N             Run up to N jobs in parallel (default: 0 = sequential)

set -e

# ─── Parse arguments ────────────────────────────────────────────────────────────
POINTS_LIST=""
SIZES_LIST=""
MAX_PARALLEL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --points)
            POINTS_LIST="$2"
            shift 2
            ;;
        --sizes)
            SIZES_LIST="$2"
            shift 2
            ;;
        --parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --points \"P1 P2 ...\" --sizes \"S1 S2 ...\" [--parallel N]"
            exit 1
            ;;
    esac
done

if [ -z "$POINTS_LIST" ] || [ -z "$SIZES_LIST" ]; then
    echo "Error: --points and --sizes are required."
    echo "Usage: $0 --points \"P1 P2 ...\" --sizes \"S1 S2 ...\" [--parallel N]"
    exit 1
fi

# Convert to arrays
read -r -a POINTS_ARR <<< "$POINTS_LIST"
read -r -a SIZES_ARR <<< "$SIZES_LIST"

TOTAL_RUNS=$(( ${#POINTS_ARR[@]} * ${#SIZES_ARR[@]} ))

echo "=========================================================="
echo " Experiment Grid"
echo "=========================================================="
echo " num_points:     ${POINTS_ARR[*]}"
echo " train_set_size: ${SIZES_ARR[*]}"
echo " Total runs:     $TOTAL_RUNS"
if [ "$MAX_PARALLEL" -gt 0 ]; then
    echo " Mode:           Parallel (max $MAX_PARALLEL concurrent)"
else
    echo " Mode:           Sequential"
fi
echo "=========================================================="
echo ""

BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

source .venv/bin/activate

# ─── Function to run a single training job ───────────────────────────────────────
run_single() {
    local pts=$1
    local sz=$2
    local job_idx=$3

    local RUN_DIR="${BASE_DIR}/pts_${pts}_train_${sz}"
    mkdir -p "$RUN_DIR"
    local LOG_FILE="${RUN_DIR}/training.log"

    echo "[Job $job_idx/$TOTAL_RUNS] Starting: num_points=$pts, train_set_size=$sz"

    PYTHONUNBUFFERED=1 python train.py \
        --num_points "$pts" \
        --epochs 200 \
        --save_dir "$RUN_DIR" \
        --train-set-size "$sz" \
        --lr 0.01 \
        --model kan_std \
        --balanced \
        > "$LOG_FILE" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[Job $job_idx/$TOTAL_RUNS] Finished: num_points=$pts, train_set_size=$sz (success)"
    else
        echo "[Job $job_idx/$TOTAL_RUNS] FAILED:   num_points=$pts, train_set_size=$sz (exit code $status)"
    fi
    return $status
}

# ─── Run the grid ────────────────────────────────────────────────────────────────
JOB_IDX=0
FAILED_JOBS=0

if [ "$MAX_PARALLEL" -le 0 ]; then
    # ── Sequential mode ──────────────────────────────────────────────────────
    for pts in "${POINTS_ARR[@]}"; do
        for sz in "${SIZES_ARR[@]}"; do
            JOB_IDX=$((JOB_IDX + 1))
            run_single "$pts" "$sz" "$JOB_IDX" || FAILED_JOBS=$((FAILED_JOBS + 1))
        done
    done
else
    # ── Parallel mode ────────────────────────────────────────────────────────
    RUNNING=0
    PIDS=()
    JOB_LABELS=()

    for pts in "${POINTS_ARR[@]}"; do
        for sz in "${SIZES_ARR[@]}"; do
            JOB_IDX=$((JOB_IDX + 1))

            # Wait if we've hit the concurrency limit
            while [ "$RUNNING" -ge "$MAX_PARALLEL" ]; do
                # Wait for any one child to finish
                wait -n 2>/dev/null || true
                # Recount how many are still running
                NEW_RUNNING=0
                NEW_PIDS=()
                NEW_LABELS=()
                for i in "${!PIDS[@]}"; do
                    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                        NEW_RUNNING=$((NEW_RUNNING + 1))
                        NEW_PIDS+=("${PIDS[$i]}")
                        NEW_LABELS+=("${JOB_LABELS[$i]}")
                    else
                        # Check exit status of finished process
                        wait "${PIDS[$i]}" 2>/dev/null || FAILED_JOBS=$((FAILED_JOBS + 1))
                    fi
                done
                PIDS=("${NEW_PIDS[@]}")
                JOB_LABELS=("${NEW_LABELS[@]}")
                RUNNING=$NEW_RUNNING
            done

            # Launch the job in the background
            run_single "$pts" "$sz" "$JOB_IDX" &
            PIDS+=($!)
            JOB_LABELS+=("pts=${pts}_sz=${sz}")
            RUNNING=$((RUNNING + 1))
        done
    done

    # Wait for all remaining jobs to finish
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" 2>/dev/null || FAILED_JOBS=$((FAILED_JOBS + 1))
    done
fi

# ─── Generate summary CSV ────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo " Generating summary..."
echo "=========================================================="

SUMMARY_FILE="${BASE_DIR}/summary.csv"
echo "num_points,train_set_size,best_test_acc,best_epoch,final_train_acc,final_test_acc" > "$SUMMARY_FILE"

for pts in "${POINTS_ARR[@]}"; do
    for sz in "${SIZES_ARR[@]}"; do
        RESULTS_FILE="${BASE_DIR}/pts_${pts}_train_${sz}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            # Extract fields from results.json using python (available since we're in a python env)
            python -c "
import json, sys
with open('$RESULTS_FILE') as f:
    r = json.load(f)
print(f\"{r['num_points']},{r['train_set_size']},{r['best_test_acc']},{r['best_epoch']},{r['final_train_acc']},{r['final_test_acc']}\")
" >> "$SUMMARY_FILE"
        else
            echo "$pts,$sz,FAILED,FAILED,FAILED,FAILED" >> "$SUMMARY_FILE"
        fi
    done
done

echo ""
echo "=========================================================="
echo " Summary saved to: $SUMMARY_FILE"
echo "=========================================================="
echo ""
cat "$SUMMARY_FILE" | column -t -s ','
echo ""

if [ "$FAILED_JOBS" -gt 0 ]; then
    echo "WARNING: $FAILED_JOBS job(s) failed. Check individual log files for details."
    exit 1
else
    echo "All $TOTAL_RUNS training runs completed successfully!"
fi
