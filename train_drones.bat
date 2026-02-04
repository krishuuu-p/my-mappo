@echo off
REM Training script for MA-LSTM-PPO on PyBullet Drones (Windows Batch)

REM Configuration
set PYTHON=C:\Users\DELL\my-mappo\.venv\Scripts\python.exe
set PYTHONPATH=C:\Users\DELL\my-mappo

REM Default parameters (can be overridden by arguments)
set NUM_DRONES=%1
set NUM_STEPS=%2
set EPISODE_LENGTH=%3
set RECURRENT_N=%4

if "%NUM_DRONES%"=="" set NUM_DRONES=3
if "%NUM_STEPS%"=="" set NUM_STEPS=5000
if "%EPISODE_LENGTH%"=="" set EPISODE_LENGTH=50
if "%RECURRENT_N%"=="" set RECURRENT_N=1

echo =========================================
echo MA-LSTM-PPO Training for PyBullet Drones
echo =========================================
echo Number of drones: %NUM_DRONES%
echo Training steps: %NUM_STEPS%
echo Episode length: %EPISODE_LENGTH%
echo Recurrent layers: %RECURRENT_N%
echo =========================================
echo.

%PYTHON% onpolicy/scripts/train/train_pybullet_drones.py ^
    --use_wandb False ^
    --num_drones %NUM_DRONES% ^
    --n_rollout_threads 1 ^
    --num_env_steps %NUM_STEPS% ^
    --episode_length %EPISODE_LENGTH% ^
    --log_interval 5 ^
    --recurrent_N %RECURRENT_N%

echo.
echo Training completed!
echo Models saved in: onpolicy\scripts\results\pybullet-drones\
pause
