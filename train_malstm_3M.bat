@echo off
REM Fix PyTorch hanging issue
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set PYTHONUNBUFFERED=1

echo Starting training with 3M steps...
"C:\Users\DELL\my-mappo\venv\Scripts\python.exe" -u onpolicy/scripts/train/train_pybullet_drones.py ^
    --num_drones 3 ^
    --num_env_steps 3000000 ^
    --n_rollout_threads 4 ^
    --n_training_threads 4 ^
    --num_mini_batch 2 ^
    --save_interval 25 ^
    --experiment_name "malstm_gpu_3M"

echo Training complete!
pause
