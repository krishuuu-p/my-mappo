"""
PyBullet Drone Runner for MA-LSTM-PPO.

This runner handles training, evaluation, and rendering for the PyBullet
drone formation flying task with MA-LSTM-PPO.

Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1 (Training loop)

Author: MA-LSTM-PPO Integration
"""

import os
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()


class PyBulletDroneRunner(Runner):
    """
    Runner for MA-LSTM-PPO training with PyBullet drones.
    
    Handles the training loop, data collection, and evaluation
    for formation flying with multiple drones.
    
    Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1 (CTDE, PPO-style)
    """
    
    def __init__(self, config):
        """
        Initialize the runner.
        
        Overrides parent to handle MA-LSTM policy selection.
        
        Args:
            config: Configuration dictionary
        """
        # Get model type from args
        all_args = config['all_args']
        model_type = getattr(all_args, 'model', 'ma_lstm')
        
        # Store for later use in policy creation
        self._model_type = model_type
        
        # Call parent init (this sets up policy, trainer, buffer)
        super(PyBulletDroneRunner, self).__init__(config)
        
        # Training history for detailed logging and plotting
        self.training_history = {
            'episodes': [],           # Episode numbers
            'timesteps': [],          # Total env steps
            'avg_rewards': [],        # Average episode rewards
            'formation_errors': [],   # Formation errors
            'avg_distances': [],      # Avg distance to target (all drones)
            'per_drone_distances': [],  # Per-drone final distances
            'per_drone_reached': [],  # Per-drone reached target (bool)
            'episode_summaries': [],  # Detailed episode summaries
        }
        
        # Track completed episodes during each training cycle
        self.recent_episode_completions = []
    
    def _init_policy(self):
        """
        Initialize policy based on model type.
        
        This method is called by parent __init__ to create the policy.
        Override to support MA-LSTM policy.
        """
        # Already handled in parent __init__ based on algorithm_name
        pass
    
    def run(self):
        """
        Main training loop.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1
        Pseudocode:
        for iteration in range(max_iters):
            for t in range(T):
                actions = policy.act(obs, rnn_states, deterministic=False)
                next_obs, rewards, dones, infos = env.step(actions)
                buffer.add(obs, actions, rewards, dones, rnn_states, values, logp)
            advantages, returns = compute_gae(buffer)
            for epoch in range(K):
                for minibatch in buffer.minibatches():
                    loss = ppo_loss(minibatch, clip=eps)
                    optimizer.step(loss)
        """
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            # Learning rate decay
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            # Collect rollouts
            # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1
            for step in range(self.episode_length):
                # Sample actions from policy
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = \
                    self.collect(step)
                
                # Environment step
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                
                data = (obs, share_obs, rewards, dones, infos, values, actions, 
                       action_log_probs, rnn_states, rnn_states_critic)
                
                # Insert data into buffer
                self.insert(data)
            
            # Compute returns and update network
            # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1
            self.compute()
            train_infos = self.train()
            
            # Post processing
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # Save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()
            
            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(f"\n{'='*80}")
                print(f" PyBullet Drones | Algo {self.algorithm_name} | Exp {self.experiment_name}")
                print(f" Episode {episode}/{episodes} | Steps {total_num_steps}/{self.num_env_steps} | "
                      f"FPS {int(total_num_steps / (end - start))}")
                print(f"{'='*80}")
                
                # Log environment-specific info
                env_infos = self._collect_env_infos(infos)
                
                avg_reward = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_reward
                print(f"\n  Average episode reward: {avg_reward:.4f}")
                
                # Log formation metrics if available
                formation_error = None
                if env_infos.get('formation_error'):
                    formation_error = np.mean(env_infos['formation_error'])
                    print(f"  Formation error: {formation_error:.4f}")
                
                # Collect detailed episode summaries from infos
                self._log_episode_details(infos, episode, total_num_steps, avg_reward, formation_error)
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                print(f"{'='*80}\n")
            
            # Evaluation
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        
        # End of training: save history and generate plots
        self._save_training_history()
        self._generate_training_plots()
    
    def warmup(self):
        """
        Initialize buffer with first observation.
        
        Reset environments and store initial observations.
        """
        # Reset env and get initial observations
        obs, share_obs, available_actions = self.envs.reset()
        
        # Handle observation shapes
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if not isinstance(share_obs, np.ndarray):
            share_obs = np.array(share_obs)
        
        # Ensure correct shape: (n_rollout_threads, num_agents, obs_dim)
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
        if len(share_obs.shape) == 2:
            share_obs = np.expand_dims(share_obs, 0)
        
        # Store in buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
    
    @torch.no_grad()
    def collect(self, step):
        """
        Collect experience for one step.
        
        Sample actions from policy and prepare for environment step.
        
        Args:
            step: Current step in episode
            
        Returns:
            Tuple of (values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env)
        """
        self.trainer.prep_rollout()
        
        # Get actions from policy
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step])
            )
        
        # Split by environment
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        
        # RNN states need special handling
        rnn_states_np = _t2n(rnn_states)
        rnn_states_critic_np = _t2n(rnn_states_critic)
        
        # Policy returns shape: (recurrent_N, n_rollout * n_agents, hidden_size)
        # or with batching: (recurrent_N, n_rollout, n_agents, hidden_size)
        # Buffer expects: (n_rollout, n_agents, recurrent_N, hidden_size)
        
        if len(rnn_states_np.shape) == 4:
            # Shape is (recurrent_N, n_rollout, n_agents, hidden) -> transpose to (n_rollout, n_agents, recurrent_N, hidden)
            rnn_states = rnn_states_np.transpose(1, 2, 0, 3)
            rnn_states_critic = rnn_states_critic_np.transpose(1, 2, 0, 3)
        elif len(rnn_states_np.shape) == 3:
            # Shape is (recurrent_N, batch, hidden) -> reshape and transpose
            # batch = n_rollout * n_agents
            batch_size = rnn_states_np.shape[1]
            rnn_states_reshaped = rnn_states_np.reshape(self.recurrent_N, self.n_rollout_threads, self.num_agents, self.hidden_size)
            rnn_states_critic_reshaped = rnn_states_critic_np.reshape(self.recurrent_N, self.n_rollout_threads, self.num_agents, self.hidden_size)
            rnn_states = rnn_states_reshaped.transpose(1, 2, 0, 3)
            rnn_states_critic = rnn_states_critic_reshaped.transpose(1, 2, 0, 3)
        else:
            raise ValueError(f"Unexpected rnn_states shape: {rnn_states_np.shape}")
        
        # For continuous action space (PyBullet drones), pass actions directly
        # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 3
        # Actions are v_des = [vx, vy, vz, throttle] in [-1, 1]
        actions_env = actions.copy()
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
    
    def insert(self, data):
        """
        Insert collected data into buffer.
        
        Args:
            data: Tuple of (obs, share_obs, rewards, dones, infos, values, 
                           actions, action_log_probs, rnn_states, rnn_states_critic)
        """
        obs, share_obs, rewards, dones, infos, values, actions, \
            action_log_probs, rnn_states, rnn_states_critic = data
        
        # Convert to numpy if needed
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if not isinstance(share_obs, np.ndarray):
            share_obs = np.array(share_obs)
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        if not isinstance(dones, np.ndarray):
            dones = np.array(dones)
        
        # Handle shapes
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
        if len(share_obs.shape) == 2:
            share_obs = np.expand_dims(share_obs, 0)
        if len(rewards.shape) == 2:
            rewards = np.expand_dims(rewards, 0)
        if len(dones.shape) == 1:
            dones = np.expand_dims(dones, 0)
        if len(dones.shape) == 2:
            dones = np.expand_dims(dones, -1)
        
        # Reset RNN states where episode ended
        done_mask = dones.squeeze(-1) == True
        for i in range(self.n_rollout_threads):
            for j in range(self.num_agents):
                if done_mask[i, j]:
                    rnn_states[i, j] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    rnn_states_critic[i, j] = np.zeros(self.buffer.rnn_states_critic.shape[3:], dtype=np.float32)
                    
                    # Capture episode summary when episode completes
                    if isinstance(infos, (list, tuple)) and i < len(infos):
                        env_info = infos[i]
                        if isinstance(env_info, (list, tuple)) and j < len(env_info):
                            agent_info = env_info[j]
                            if isinstance(agent_info, dict) and 'episode_summary' in agent_info:
                                self.recent_episode_completions.append({
                                    'env_idx': i,
                                    'agent_id': j,
                                    'summary': agent_info['episode_summary']
                                })
        
        # Create masks
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[done_mask] = 0.0
        
        # Insert into buffer
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, 
                          actions, action_log_probs, values, rewards, masks)
    
    def _collect_env_infos(self, infos):
        """
        Collect environment-specific information for logging.
        
        Args:
            infos: List of info dicts from environment
            
        Returns:
            Dict of aggregated info
        """
        env_infos = {}
        
        # Collect individual rewards
        for agent_id in range(self.num_agents):
            idv_rews = []
            form_errors = []
            for info in infos:
                if isinstance(info, dict):
                    if 'individual_reward' in info:
                        idv_rews.append(info['individual_reward'])
                    if 'formation_error' in info:
                        form_errors.append(info['formation_error'])
                elif isinstance(info, list) and len(info) > agent_id:
                    if 'individual_reward' in info[agent_id]:
                        idv_rews.append(info[agent_id]['individual_reward'])
                    if 'formation_error' in info[agent_id]:
                        form_errors.append(info[agent_id]['formation_error'])
            
            if idv_rews:
                env_infos[f'agent{agent_id}/individual_rewards'] = idv_rews
            if form_errors:
                env_infos[f'agent{agent_id}/formation_error'] = form_errors
        
        # Aggregate formation error
        if env_infos.get('agent0/formation_error'):
            env_infos['formation_error'] = env_infos['agent0/formation_error']
        
        return env_infos
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        """
        Evaluate the current policy.
        
        Args:
            total_num_steps: Current total training steps (for logging)
        """
        eval_episode_rewards = []
        eval_obs, eval_share_obs = self.eval_envs.reset()
        
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), 
            dtype=np.float32
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), 
            dtype=np.float32
        )
        
        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            
            # Get deterministic actions
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Step environment
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = \
                self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)
            
            # Reset RNN states for done episodes
            eval_dones_arr = np.array(eval_dones)
            if len(eval_dones_arr.shape) == 1:
                eval_dones_arr = np.expand_dims(eval_dones_arr, 0)
            
            eval_rnn_states[eval_dones_arr == True] = np.zeros(
                ((eval_dones_arr == True).sum(), self.recurrent_N, self.hidden_size), 
                dtype=np.float32
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), 
                dtype=np.float32
            )
            eval_masks[eval_dones_arr == True] = np.zeros(
                ((eval_dones_arr == True).sum(), 1), 
                dtype=np.float32
            )
        
        # Log evaluation results
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {
            'eval_average_episode_rewards': np.sum(eval_episode_rewards, axis=0)
        }
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print(f"Eval average episode rewards: {eval_average_episode_rewards}")
        self.log_env(eval_env_infos, total_num_steps)
    
    @torch.no_grad()
    def render(self):
        """
        Render episodes with the trained policy.
        Loops continuously until user closes the window or presses Ctrl+C.
        """
        envs = self.envs
        
        # Create gif directory if saving gifs
        if self.all_args.save_gifs:
            print("\n[INFO] GIF generation is not currently supported for PyBullet drones.")
            print("[INFO] The PyBullet environment's render() method returns text output only.")
            print("[INFO] To visualize your trained model, run without --save_gifs flag.")
            print("[INFO] The GUI window will show the drones flying in real-time.\n")
            return
        
        print("\n[INFO] Starting visualization. Press Ctrl+C to stop and close the window.\n")
        
        all_frames = []
        episode_count = 0
        
        try:
            # Loop indefinitely until user interrupts
            while True:
                # Run the specified number of episodes, then repeat
                for episode in range(self.all_args.render_episodes):
                    episode_num = episode_count + episode
                    obs, share_obs, available_actions = envs.reset()
                    
                    # Draw visual markers for initial and target positions
                    try:
                        # Access the actual environment from the vectorized wrapper
                        actual_env = envs.envs[0] if hasattr(envs, 'envs') else envs
                        actual_env.draw_position_markers()
                        print(f"\n[Episode {episode_num}] Visual markers drawn:")
                        print(f"  Green spheres = Initial positions")
                        print(f"  Red spheres = Target positions")
                        print(f"  Gray lines = Initial → Target path\n")
                    except Exception as e:
                        print(f"[DEBUG] Could not draw markers: {e}")
                    
                    if self.all_args.save_gifs:
                        try:
                            render_output = envs.render('rgb_array')
                            if render_output is not None:
                                image = render_output[0][0] if isinstance(render_output, list) else render_output
                                all_frames.append(image)
                        except Exception as e:
                            print(f"Warning: Failed to capture initial frame for episode {episode_num}: {e}")
                    else:
                        envs.render('human')
                    
                    rnn_states = np.zeros(
                        (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), 
                        dtype=np.float32
                    )
                    masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    
                    episode_rewards = []
                    
                    for step in range(self.episode_length):
                        calc_start = time.time()
                        
                        self.trainer.prep_rollout()
                        action, rnn_states = self.trainer.policy.act(
                            np.concatenate(obs),
                            np.concatenate(rnn_states),
                            np.concatenate(masks),
                            deterministic=True
                        )
                        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                        
                        # Step environment
                        obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
                        episode_rewards.append(rewards)
                        
                        # Reset RNN states for done episodes
                        dones_arr = np.array(dones)
                        if len(dones_arr.shape) == 1:
                            dones_arr = np.expand_dims(dones_arr, 0)
                        
                        rnn_states[dones_arr == True] = np.zeros(
                            ((dones_arr == True).sum(), self.recurrent_N, self.hidden_size), 
                            dtype=np.float32
                        )
                        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                        masks[dones_arr == True] = np.zeros(((dones_arr == True).sum(), 1), dtype=np.float32)
                        
                        if self.all_args.save_gifs:
                            try:
                                render_output = envs.render('rgb_array')
                                if render_output is not None:
                                    image = render_output[0][0] if isinstance(render_output, list) else render_output
                                    all_frames.append(image)
                            except Exception as e:
                                print(f"Warning: Failed to capture frame at step {step}: {e}")
                            calc_end = time.time()
                            elapsed = calc_end - calc_start
                            if elapsed < self.all_args.ifi:
                                time.sleep(self.all_args.ifi - elapsed)
                        else:
                            envs.render('human')
                    
                    print(f"Episode {episode_num} average rewards: {np.mean(np.sum(np.array(episode_rewards), axis=0))}")
                
                episode_count += self.all_args.render_episodes
                
        except KeyboardInterrupt:
            print(f"\n\n[INFO] Visualization stopped by user after {episode_count} episodes.")
            print("[INFO] Closing PyBullet window...")
    
    def _log_episode_details(self, infos, episode, total_num_steps, avg_reward, formation_error):
        """
        Log detailed episode information including positions, distances, and directions.
        
        Args:
            infos: Info dicts from the last step (tuple of env infos across rollout threads)
            episode: Current episode number
            total_num_steps: Total training steps so far
            avg_reward: Average reward for this episode
            formation_error: Formation error (or None)
        """
        # First check if we have any episode completions from this rollout period
        episode_details_to_show = None
        
        if self.recent_episode_completions:
            # Group completions by environment
            env_completions = {}
            for completion in self.recent_episode_completions:
                env_idx = completion['env_idx']
                if env_idx not in env_completions:
                    env_completions[env_idx] = []
                env_completions[env_idx].append(completion)
            
            # Use completions from environment 0 if available
            # Only show the most recent episode (last num_agents completions)
            if 0 in env_completions:
                recent_completions = env_completions[0][-self.num_agents:]
                episode_details_to_show = {
                    'env_idx': 0,
                    'drones': [
                        {
                            'drone_id': c['agent_id'],
                            **c['summary']
                        }
                        for c in sorted(recent_completions, key=lambda x: x['agent_id'])
                    ]
                }
            
            # Clear the completions for next logging period
            self.recent_episode_completions.clear()
        
        # Collect episode summaries from all rollout threads
        all_final_distances = []
        all_reached = []
        episode_details = []
        current_positions_log = []  # For logging current positions even if episode not done
        
        for env_idx, info in enumerate(infos):
            if not isinstance(info, (list, tuple)):
                continue
            
            env_summary = {'env_idx': env_idx, 'drones': []}
            env_current = {'env_idx': env_idx, 'drones': []}
            has_summary = False
            
            for agent_id in range(min(len(info), self.num_agents)):
                agent_info = info[agent_id] if isinstance(info[agent_id], dict) else {}
                
                # Get current distances
                dist = agent_info.get('dist_to_target', None)
                if dist is not None:
                    all_final_distances.append(dist)
                
                # Try to extract current position info from agent_info first
                current_pos_info = agent_info.get('current_pos', None)
                target_pos_info = agent_info.get('target_pos', None)
                
                if current_pos_info is not None and target_pos_info is not None:
                    # Got positions directly from info
                    env_current['drones'].append({
                        'drone_id': agent_id,
                        'current_position': list(current_pos_info),
                        'target_position': list(target_pos_info),
                        'distance': dist if dist is not None else np.linalg.norm(np.array(target_pos_info) - np.array(current_pos_info)),
                    })
                else:
                    # Extract current position and target from observation (available every step)
                    # Observation structure: [own_pos(3), own_vel(3), rel_target(3), neighbors...]
                    # Get the last observation for this agent from buffer
                    if hasattr(self, 'buffer') and self.buffer.obs is not None:
                        try:
                            # Buffer shape: [episode_length+1, n_rollout_threads, num_agents, obs_dim]
                            # Access the most recent complete observation (step -2 before warmup overwrites)
                            obs_step = min(self.episode_length - 1, self.buffer.obs.shape[0] - 1)
                            current_obs = self.buffer.obs[obs_step][env_idx][agent_id]
                            current_pos = current_obs[0:3]  # First 3 elements are position
                            rel_target = current_obs[6:9]   # Elements 6-8 are relative target
                            target_pos = current_pos + rel_target
                            
                            env_current['drones'].append({
                                'drone_id': agent_id,
                                'current_position': current_pos.tolist() if hasattr(current_pos, 'tolist') else list(current_pos),
                                'target_position': target_pos.tolist() if hasattr(target_pos, 'tolist') else list(target_pos),
                                'distance': dist if dist is not None else np.linalg.norm(rel_target),
                            })
                        except (IndexError, AttributeError, TypeError) as e:
                            # Fallback: try to construct from agent_info if available
                            pass
                
                # Check for episode summary (only present on done steps)
                summary = agent_info.get('episode_summary', None)
                if summary is not None:
                    has_summary = True
                    all_reached.append(summary['reached_target'])
                    env_summary['drones'].append({
                        'drone_id': agent_id,
                        **summary,
                        'episode_steps': agent_info.get('episode_steps', 0),
                        'episode_reward': agent_info.get('episode_cumulative_reward', 0),
                    })
            
            if has_summary:
                episode_details.append(env_summary)
            if env_current['drones']:
                current_positions_log.append(env_current)
        
        # Compute aggregate metrics
        # Prioritize episode_details_to_show if available (from accumulated recent completions)
        if episode_details_to_show and episode_details_to_show['drones']:
            # Use accumulated completions as source of truth - replace previous data
            all_reached = []
            all_final_distances = []
            for drone in episode_details_to_show['drones']:
                all_reached.append(drone.get('reached_target', False))
                all_final_distances.append(drone.get('final_distance', 0))
        elif episode_details:
            # Use data from current step's infos (already populated from loop above)
            pass
        
        avg_distance = np.mean(all_final_distances) if all_final_distances else None
        num_reached = sum(all_reached) if all_reached else 0
        total_drones = len(all_reached) if all_reached else 0
        
        # Store in history
        self.training_history['episodes'].append(episode)
        self.training_history['timesteps'].append(total_num_steps)
        self.training_history['avg_rewards'].append(avg_reward)
        self.training_history['formation_errors'].append(formation_error)
        self.training_history['avg_distances'].append(avg_distance)
        self.training_history['per_drone_distances'].append(all_final_distances)
        self.training_history['per_drone_reached'].append(all_reached)
        # Store episode_details_to_show if available, otherwise episode_details
        self.training_history['episode_summaries'].append([episode_details_to_show] if episode_details_to_show else episode_details)
        
        # Print detailed per-drone info
        if avg_distance is not None:
            print(f"  Avg distance to target: {avg_distance:.4f}m")
        if total_drones > 0:
            print(f"  Targets reached: {num_reached}/{total_drones} drones "
                  f"({100*num_reached/total_drones:.1f}%)")
        
        # Always print positions for the first environment every 5th episode
        if episode_details_to_show:
            # Episode completed during this rollout - show initial, final, and target
            env_idx = episode_details_to_show['env_idx']
            print(f"\n  --- Episode Completed in Env {env_idx} ---")
            for drone in episode_details_to_show['drones']:
                d = drone
                init = d['initial_position']
                final = d['final_position']
                target = d['target_position']
                print(f"    Drone {d['drone_id']}:")
                print(f"      Started at:       [{init[0]:+.2f}, {init[1]:+.2f}, {init[2]:+.2f}]")
                print(f"      Ended at:         [{final[0]:+.2f}, {final[1]:+.2f}, {final[2]:+.2f}]")
                print(f"      Should reach:     [{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]")
                status = 'REACHED ✓' if d['reached_target'] else f"Distance remaining: {d['final_distance']:.3f}m"
                print(f"      Status: {status}")
        elif episode_details:
            # Episode completed at the last step - show initial, final, and target
            env_summary = episode_details[0]
            env_idx = env_summary['env_idx']
            print(f"\n  --- Episode Completed in Env {env_idx} ---")
            for drone in env_summary['drones']:
                d = drone
                init = d['initial_position']
                final = d['final_position']
                target = d['target_position']
                print(f"    Drone {d['drone_id']}:")
                print(f"      Started at:       [{init[0]:+.2f}, {init[1]:+.2f}, {init[2]:+.2f}]")
                print(f"      Ended at:         [{final[0]:+.2f}, {final[1]:+.2f}, {final[2]:+.2f}]")
                print(f"      Should reach:     [{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]")
                status = 'REACHED ✓' if d['reached_target'] else f"Distance remaining: {d['final_distance']:.3f}m"
                print(f"      Status: {status}")
        elif current_positions_log:
            # Episode in progress - show current position and target
            env_current = current_positions_log[0]
            env_idx = env_current['env_idx']
            print(f"\n  --- Current Positions in Env {env_idx} (Episode In Progress) ---")
            for drone in env_current['drones']:
                curr = drone['current_position']
                target = drone['target_position']
                dist = drone['distance']
                print(f"    Drone {drone['drone_id']}:")
                print(f"      Current position: [{curr[0]:+.2f}, {curr[1]:+.2f}, {curr[2]:+.2f}]")
                print(f"      Should reach:     [{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]")
                print(f"      Distance to target: {dist:.3f}m")
        else:
            print(f"\n  --- No position data available for logging ---")
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        save_path = os.path.join(self.save_dir if hasattr(self, 'save_dir') else '.', 'training_history.json')
        
        # Convert numpy types to Python native types for JSON
        serializable_history = {}
        for key, values in self.training_history.items():
            if key == 'episode_summaries':
                serializable_history[key] = values  # Already serializable
            else:
                serializable_history[key] = []
                for v in values:
                    if v is None:
                        serializable_history[key].append(None)
                    elif isinstance(v, (list, tuple)):
                        serializable_history[key].append([float(x) if isinstance(x, (int, float, np.floating)) else bool(x) for x in v])
                    elif isinstance(v, (np.floating, np.integer)):
                        serializable_history[key].append(float(v))
                    else:
                        serializable_history[key].append(v)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=str)
            print(f"\n[INFO] Training history saved to: {save_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save training history: {e}")
    
    def _generate_training_plots(self):
        """Generate comprehensive training plots at the end of training."""
        plot_dir = os.path.join(self.save_dir if hasattr(self, 'save_dir') else '.', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        history = self.training_history
        if len(history['episodes']) < 2:
            print("[INFO] Not enough data points for plotting.")
            return
        
        episodes = history['episodes']
        timesteps = history['timesteps']
        
        # ---- Figure 1: Rewards over time ----
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Summary: {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # 1a: Average Episode Reward
        ax = axes[0, 0]
        rewards = history['avg_rewards']
        ax.plot(timesteps, rewards, 'b-', alpha=0.3, linewidth=0.8)
        # Smoothed curve (moving average)
        window = max(1, len(rewards) // 20)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(timesteps[window-1:], smoothed, 'b-', linewidth=2, label=f'Smoothed (w={window})')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Average Episode Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1b: Formation Error
        ax = axes[0, 1]
        form_errors = [fe if fe is not None else np.nan for fe in history['formation_errors']]
        ax.plot(timesteps, form_errors, 'r-', alpha=0.3, linewidth=0.8)
        if len(form_errors) >= window:
            valid_fe = np.array(form_errors, dtype=float)
            # Handle NaNs for smoothing
            mask = ~np.isnan(valid_fe)
            if mask.sum() >= window:
                smoothed_fe = np.convolve(valid_fe[mask], np.ones(window)/window, mode='valid')
                ax.plot(np.array(timesteps)[mask][window-1:], smoothed_fe, 'r-', linewidth=2, label=f'Smoothed')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Formation Error (E)')
        ax.set_title('Formation Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1c: Average Distance to Target
        ax = axes[1, 0]
        avg_dists = [d if d is not None else np.nan for d in history['avg_distances']]
        ax.plot(timesteps, avg_dists, 'g-', alpha=0.3, linewidth=0.8)
        if len(avg_dists) >= window:
            valid_d = np.array(avg_dists, dtype=float)
            mask = ~np.isnan(valid_d)
            if mask.sum() >= window:
                smoothed_d = np.convolve(valid_d[mask], np.ones(window)/window, mode='valid')
                ax.plot(np.array(timesteps)[mask][window-1:], smoothed_d, 'g-', linewidth=2, label='Smoothed')
        ax.axhline(y=0.05, color='k', linestyle='--', alpha=0.5, label='Goal threshold (5cm)')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Avg Distance to Target (m)')
        ax.set_title('Distance to Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1d: Target Reached Rate
        ax = axes[1, 1]
        reached_rates = []
        for reached_list in history['per_drone_reached']:
            if reached_list and len(reached_list) > 0:
                reached_rates.append(100 * sum(reached_list) / len(reached_list))
            else:
                reached_rates.append(np.nan)
        ax.plot(timesteps, reached_rates, 'm-', alpha=0.3, linewidth=0.8)
        if len(reached_rates) >= window:
            valid_r = np.array(reached_rates, dtype=float)
            mask = ~np.isnan(valid_r)
            if mask.sum() >= window:
                smoothed_r = np.convolve(valid_r[mask], np.ones(window)/window, mode='valid')
                ax.plot(np.array(timesteps)[mask][window-1:], smoothed_r, 'm-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Target Reached Rate (%)')
        ax.set_title('Drones Reaching Target (<5cm)')
        ax.set_ylim(-5, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(plot_dir, 'training_summary.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Training summary plot saved to: {fig_path}")
        
        # ---- Figure 2: Per-drone distance box plots at intervals ----
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
        # Sample 10 evenly spaced episodes for box plot
        n_samples = min(10, len(history['per_drone_distances']))
        sample_indices = np.linspace(0, len(history['per_drone_distances'])-1, n_samples, dtype=int)
        box_data = []
        box_labels = []
        for idx in sample_indices:
            dists = history['per_drone_distances'][idx]
            if dists and len(dists) > 0:
                box_data.append(dists)
                ep = history['episodes'][idx]
                box_labels.append(f'Ep {ep}')
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Goal threshold (5cm)')
            ax2.set_xlabel('Training Episode')
            ax2.set_ylabel('Distance to Target (m)')
            ax2.set_title('Per-Drone Distance Distribution Over Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2_path = os.path.join(plot_dir, 'distance_distribution.png')
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"[INFO] Distance distribution plot saved to: {fig2_path}")
        
        # ---- Figure 3: Reward components breakdown (from last logged episodes) ----
        # Extract reward components from episode summaries
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig3.suptitle('Reward Components Over Training', fontsize=14, fontweight='bold')
        
        r_forms, r_navs, r_avoids = [], [], []
        valid_ts = []
        for idx, summaries in enumerate(history['episode_summaries']):
            if summaries:
                for env_sum in summaries:
                    for drone in env_sum.get('drones', []):
                        # These come from the info dict
                        pass
            # Use the formation error as proxy for r_form
            fe = history['formation_errors'][idx]
            if fe is not None:
                r_forms.append(-fe)
                valid_ts.append(timesteps[idx])
        
        if r_forms:
            ax = axes3[0]
            ax.plot(valid_ts, r_forms, 'b-', alpha=0.5)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('r_form (-E/(G+eps))')
            ax.set_title('Formation Reward')
            ax.grid(True, alpha=0.3)
        
        # Distance improvement over time
        ax = axes3[1]
        dist_improvements = []
        dist_ts = []
        for idx, summaries in enumerate(history['episode_summaries']):
            improvements = []
            for env_sum in summaries:
                for drone in env_sum.get('drones', []):
                    improvements.append(drone.get('distance_improvement', 0))
            if improvements:
                dist_improvements.append(np.mean(improvements))
                dist_ts.append(timesteps[idx])
        if dist_improvements:
            ax.plot(dist_ts, dist_improvements, 'g-', alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Avg Distance Improvement (m)')
            ax.set_title('Navigation Progress')
            ax.grid(True, alpha=0.3)
        
        # Direction accuracy over time
        ax = axes3[2]
        direction_angles = []
        angle_ts = []
        for idx, summaries in enumerate(history['episode_summaries']):
            angles = []
            for env_sum in summaries:
                for drone in env_sum.get('drones', []):
                    traveled = np.array(drone.get('direction_traveled', [0,0,0]))
                    needed = np.array(drone.get('direction_to_target', [0,0,0]))
                    if np.linalg.norm(needed) > 1e-6 and np.linalg.norm(traveled) > 1e-6:
                        cos_a = np.dot(traveled, needed) / (np.linalg.norm(traveled) * np.linalg.norm(needed))
                        cos_a = np.clip(cos_a, -1.0, 1.0)
                        angles.append(np.degrees(np.arccos(cos_a)))
            if angles:
                direction_angles.append(np.mean(angles))
                angle_ts.append(timesteps[idx])
        if direction_angles:
            ax.plot(angle_ts, direction_angles, 'orange', alpha=0.5)
            ax.axhline(y=0, color='g', linestyle='--', alpha=0.3, label='Perfect direction')
            ax.axhline(y=90, color='r', linestyle='--', alpha=0.3, label='Perpendicular')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Direction Error (degrees)')
            ax.set_title('Direction Accuracy')
            ax.set_ylim(-5, 185)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig3_path = os.path.join(plot_dir, 'reward_components.png')
        fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"[INFO] Reward components plot saved to: {fig3_path}")
        
        print(f"\n[INFO] All training plots saved to: {plot_dir}/")


