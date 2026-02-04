"""
PyBullet Drone Runner for MA-LSTM-PPO.

This runner handles training, evaluation, and rendering for the PyBullet
drone formation flying task with MA-LSTM-PPO.

Reference: docs/MA-LSTM-PPO-paper-summary.md Section 1 (Training loop)

Author: MA-LSTM-PPO Integration
"""

import time
import numpy as np
import torch
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
                print(f"\n PyBullet Drones Algo {self.algorithm_name} Exp {self.experiment_name} "
                      f"updates {episode}/{episodes} episodes, "
                      f"total num timesteps {total_num_steps}/{self.num_env_steps}, "
                      f"FPS {int(total_num_steps / (end - start))}.\n")
                
                # Log environment-specific info
                env_infos = self._collect_env_infos(infos)
                
                train_infos["average_episode_rewards"] = \
                    np.mean(self.buffer.rewards) * self.episode_length
                print(f"Average episode rewards: {train_infos['average_episode_rewards']}")
                
                # Log formation metrics if available
                if env_infos.get('formation_error'):
                    print(f"Formation error: {np.mean(env_infos['formation_error']):.4f}")
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
            
            # Evaluation
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
    
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

