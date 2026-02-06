#!/usr/bin/env python
"""
Training script for MA-LSTM-PPO with PyBullet Drones.

This script trains the MA-LSTM-PPO algorithm on the PyBullet drone formation
flying task.

Reference: docs/MA-LSTM-PPO-paper-summary.md

Usage:
    python train_pybullet_drones.py --num_drones 3 --num_env_steps 1000000

For local testing:
    python train_pybullet_drones.py --n_rollout_threads 2 --num_env_steps 10000

Author: MA-LSTM-PPO Integration
"""

import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from onpolicy.config import get_config
from onpolicy.envs.pybullet_drone_env import PyBulletDroneWrapper
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def make_train_env(all_args):
    """
    Create training environments.
    
    Args:
        all_args: Argument namespace
        
    Returns:
        Vectorized environment
    """
    def get_env_fn(rank):
        def init_env():
            env = PyBulletDroneWrapper(
                num_drones=all_args.num_drones,
                gui=False,  # No GUI for training
                use_formation_reward=True,
            )
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """
    Create evaluation environments.
    
    Args:
        all_args: Argument namespace
        
    Returns:
        Vectorized environment
    """
    def get_env_fn(rank):
        def init_env():
            env = PyBulletDroneWrapper(
                num_drones=all_args.num_drones,
                gui=False,
                use_formation_reward=True,
            )
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """
    Parse command line arguments for PyBullet drones training.
    
    Adds environment-specific arguments to the base MAPPO config.
    
    Reference: docs/MA-LSTM-PPO-paper-summary.md Section 5 (hyperparameters)
    """
    # Drone environment arguments (num_drones is already defined in config.py)
    parser.add_argument('--use_formation_reward', action='store_true', default=True,
                        help="Use formation-based reward")
    
    all_args = parser.parse_known_args(args)[0]
    
    # Set MA-LSTM model if not specified (--model already defined in config.py)
    if not hasattr(all_args, 'model') or all_args.model is None:
        all_args.model = 'ma_lstm'
    
    return all_args


def main(args):
    """Main training function."""
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # Set environment name
    all_args.env_name = "pybullet-drones"
    
    # Configure algorithm based on model type
    # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1
    if all_args.model == "ma_lstm":
        print("Using MA-LSTM-PPO model with LSTM actor and centralized critic")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        all_args.algorithm_name = "rmappo"  # Use recurrent MAPPO trainer
    else:
        print("Using standard MAPPO model")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        all_args.algorithm_name = "rmappo"
    
    # Set paper default hyperparameters if not specified
    # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 5
    if not hasattr(all_args, 'lr') or all_args.lr == 5e-4:
        all_args.lr = 5e-4  # Paper default
    if not hasattr(all_args, 'gamma') or all_args.gamma == 0.99:
        all_args.gamma = 0.99  # Paper default
    if not hasattr(all_args, 'gae_lambda') or all_args.gae_lambda == 0.95:
        all_args.gae_lambda = 0.95  # Paper default
    if not hasattr(all_args, 'clip_param') or all_args.clip_param == 0.2:
        all_args.clip_param = 0.2  # Paper default
    if not hasattr(all_args, 'episode_length'):
        all_args.episode_length = 242  # Paper suggests ~242 timesteps
    
    # CUDA setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU for training...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU for training...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # Set up run directory
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
        / all_args.env_name / f"drones_{all_args.num_drones}" / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # Wandb or tensorboard logging
    if all_args.use_wandb:
        import wandb
        run = wandb.init(
            config=all_args,
            project="pybullet-drones",
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.model}_{all_args.experiment_name}_seed{all_args.seed}",
            group=f"drones_{all_args.num_drones}",
            dir=str(run_dir),
            job_type="training",
            reinit=True
        )
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) 
                            for folder in run_dir.iterdir() 
                            if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = f'run{max(exst_run_nums) + 1}'
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    setproctitle.setproctitle(
        f"{all_args.model}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )
    
    # Set random seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    # Create environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_drones
    
    # Configuration for runner
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    # Import and create runner
    # Use shared runner for MA-LSTM (all agents share policy)
    from onpolicy.runner.shared.pybullet_drone_runner import PyBulletDroneRunner as Runner
    
    runner = Runner(config)
    runner.run()
    
    # Cleanup
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
