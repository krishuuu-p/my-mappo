#!/bin/bash

# Training script for MA-LSTM-PPO with PyBullet Drones
#
# Reference: docs/MA-LSTM-PPO-paper-summary.md Section 5 (hyperparameters)
#
# Paper default hyperparameters:
# - lr = 5e-4
# - gamma = 0.99
# - gae_lambda = 0.95
# - ppo_clip = 0.2
# - episode_length ~= 242 timesteps
#
# For local testing (fast):
#   ./train_malstm_pybullet.sh --n_rollout_threads 2 --num_env_steps 10000
#
# For full training:
#   ./train_malstm_pybullet.sh

# Default parameters
NUM_DRONES=${NUM_DRONES:-3}
SEED=${SEED:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"ma_lstm_formation"}

# Paper-recommended hyperparameters
LR=5e-4
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_PARAM=0.2
EPISODE_LENGTH=242
PPO_EPOCH=15
NUM_MINI_BATCH=1
HIDDEN_SIZE=64

# Training configuration
# Default: full training
# For testing: override with --num_env_steps 10000 --n_rollout_threads 2
NUM_ENV_STEPS=${NUM_ENV_STEPS:-1000000}
N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS:-8}

# Navigate to scripts directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."

echo "====================================="
echo "MA-LSTM-PPO Training for PyBullet Drones"
echo "====================================="
echo "Number of drones: $NUM_DRONES"
echo "Number of env steps: $NUM_ENV_STEPS"
echo "Rollout threads: $N_ROLLOUT_THREADS"
echo "Seed: $SEED"
echo "====================================="

python -m onpolicy.scripts.train.train_pybullet_drones \
    --env_name "pybullet-drones" \
    --algorithm_name "rmappo" \
    --model "ma_lstm" \
    --experiment_name "$EXPERIMENT_NAME" \
    --num_drones $NUM_DRONES \
    --seed $SEED \
    --n_training_threads 1 \
    --n_rollout_threads $N_ROLLOUT_THREADS \
    --num_env_steps $NUM_ENV_STEPS \
    --episode_length $EPISODE_LENGTH \
    --lr $LR \
    --critic_lr $LR \
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
    --clip_param $CLIP_PARAM \
    --ppo_epoch $PPO_EPOCH \
    --num_mini_batch $NUM_MINI_BATCH \
    --hidden_size $HIDDEN_SIZE \
    --use_recurrent_policy \
    --recurrent_N 1 \
    --data_chunk_length 10 \
    --use_valuenorm \
    --use_ReLU \
    --use_orthogonal \
    --log_interval 5 \
    --save_interval 10 \
    --use_eval \
    --eval_interval 25 \
    --eval_episodes 3 \
    "$@"

echo "Training completed!"
