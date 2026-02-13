#!/bin/bash
# Training script for MA-LSTM-PPO on PyBullet Drones

# Configuration
PYTHON="C:/Users/DELL/my-mappo/.venv/Scripts/python.exe"
export PYTHONPATH="C:\Users\DELL\my-mappo"

# Default parameters
NUM_DRONES=${1:-3}
NUM_STEPS=${2:-1000000}
EPISODE_LENGTH=${3:-384}
RECURRENT_N=${4:-1}
N_ROLLOUT_THREADS=${5:-4}
LOG_INTERVAL=${6:-10}

echo "========================================="
echo "MA-LSTM-PPO Training for PyBullet Drones"
echo "========================================="
echo "Number of drones: $NUM_DRONES"
echo "Training steps: $NUM_STEPS"
echo "Episode length: $EPISODE_LENGTH"
echo "Recurrent layers: $RECURRENT_N"
echo "Rollout threads: $N_ROLLOUT_THREADS"
echo "Log interval: $LOG_INTERVAL"
echo "========================================="
echo ""

$PYTHON onpolicy/scripts/train/train_pybullet_drones.py \
    --use_wandb False \
    --num_drones $NUM_DRONES \
    --n_rollout_threads $N_ROLLOUT_THREADS \
    --num_env_steps $NUM_STEPS \
    --episode_length $EPISODE_LENGTH \
    --log_interval $LOG_INTERVAL \
    --recurrent_N $RECURRENT_N

echo ""
echo "Training completed!"
echo "Models saved in: onpolicy/scripts/results/pybullet-drones/"
