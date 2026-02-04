# Makefile for MA-LSTM-PPO PyBullet Drones Training and Visualization

# Python environment
PYTHON := C:/Users/DELL/my-mappo/.venv/Scripts/python.exe
PYTHONPATH := C:\Users\DELL\my-mappo

# Default parameters
NUM_DRONES := 3
NUM_STEPS := 5000
EPISODE_LENGTH := 50
RECURRENT_N := 1
N_ROLLOUT_THREADS := 1
LOG_INTERVAL := 5

# Model directory (will be auto-generated, but you can override)
MODEL_DIR := C:\Users\DELL\my-mappo\onpolicy\scripts\results\pybullet-drones\drones_3\rmappo\check\run26\models

.PHONY: help train render clean

help:
	@echo "Available commands:"
	@echo "  make train          - Train MA-LSTM-PPO model on PyBullet drones"
	@echo "  make render         - Visualize trained model (loops continuously)"
	@echo "  make clean          - Remove training results"
	@echo ""
	@echo "Parameters (can be overridden):"
	@echo "  NUM_DRONES=$(NUM_DRONES)"
	@echo "  NUM_STEPS=$(NUM_STEPS)"
	@echo "  EPISODE_LENGTH=$(EPISODE_LENGTH)"
	@echo "  RECURRENT_N=$(RECURRENT_N)"
	@echo ""
	@echo "Example with custom parameters:"
	@echo "  make train NUM_STEPS=10000 RECURRENT_N=2"

train:
	@echo "Starting MA-LSTM-PPO training..."
	@echo "Parameters: $(NUM_DRONES) drones, $(NUM_STEPS) steps, episode length $(EPISODE_LENGTH)"
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) onpolicy/scripts/train/train_pybullet_drones.py \
		--use_wandb False \
		--num_drones $(NUM_DRONES) \
		--n_rollout_threads $(N_ROLLOUT_THREADS) \
		--num_env_steps $(NUM_STEPS) \
		--episode_length $(EPISODE_LENGTH) \
		--log_interval $(LOG_INTERVAL) \
		--recurrent_N $(RECURRENT_N)

render:
	@echo "Starting visualization (Press Ctrl+C to stop)..."
	@echo "Model directory: $(MODEL_DIR)"
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) onpolicy/scripts/render/render_pybullet_drones.py \
		--model ma_lstm \
		--use_render \
		--model_dir "$(MODEL_DIR)" \
		--num_drones $(NUM_DRONES) \
		--n_rollout_threads $(N_ROLLOUT_THREADS) \
		--episode_length $(EPISODE_LENGTH) \
		--render_episodes 3 \
		--recurrent_N $(RECURRENT_N)

clean:
	@echo "Removing training results..."
	@if exist onpolicy\scripts\results rmdir /s /q onpolicy\scripts\results
	@echo "Done!"
