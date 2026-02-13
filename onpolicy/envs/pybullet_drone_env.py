"""
PyBullet Drone Environment Wrapper for MAPPO.

This module wraps gym-pybullet-drones MultiHoverAviary to provide a MAPPO-compatible
interface with proper observation/share_obs structure.

Reference: docs/MA-LSTM-PPO-paper-summary.md Section 6 (Implementation mapping)
- reset() -> (per_agent_obs, share_obs)
- step(actions) -> (per_agent_obs, share_obs, rewards, dones, infos)
- share_obs: concatenation of all agents' positions and velocities, tiled per agent

Author: MA-LSTM-PPO Integration
"""

import numpy as np
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional

# Import from local gym_pybullet_drones copy
from onpolicy.envs.gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from onpolicy.envs.gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Formation utilities for Procrustes/SVD-based reward (paper Section 4)
from onpolicy.utils.formation import compute_formation_error


class PyBulletDroneWrapper:
    """
    MAPPO-compatible wrapper for gym-pybullet-drones MultiHoverAviary.
    
    This wrapper:
    1. Converts the gym-pybullet-drones interface to MAPPO's expected format
    2. Constructs share_obs as concatenated positions+velocities of all agents (tiled)
    3. Computes formation-based rewards as per MA-LSTM-PPO paper
    
    Reference: docs/MA-LSTM-PPO-paper-summary.md Sections 4, 6
    """

    def __init__(
        self,
        num_drones: int = 3,
        gui: bool = False,
        record: bool = False,
        obs_type: ObservationType = ObservationType.KIN,
        act_type: ActionType = ActionType.VEL,  # VEL uses internal PID, accepts v_des
        pyb_freq: int = 240,
        ctrl_freq: int = 30,  # Paper default: 30Hz → episode_length=8*30=240≈242 (Table 3)
        episode_len_sec: float = 8.0,
        collision_dist: float = 0.1,
        w_form: float = 0.1,
        w_nav: float = 5.0,
        w_dist: float = 1.0,
        w_avoid: float = 1.0,
        collision_C: float = 1.0,
    ):
        """
        Initialize the PyBullet drone wrapper.
        
        Args:
            num_drones: Number of drones in the environment
            gui: Whether to use PyBullet GUI
            record: Whether to record video
            obs_type: Observation type (KIN for kinematic)
            act_type: Action type (VEL for velocity-based with internal PID)
            pyb_freq: PyBullet simulation frequency
            ctrl_freq: Control frequency (48Hz to match simulation)
            episode_len_sec: Episode length in seconds
            collision_dist: Distance threshold for collision detection
            w_form: Weight for formation reward (downweighted for navigation tasks)
            w_nav: Weight for navigation reward — primary signal (paper Section 4)
            w_dist: Weight for persistent distance penalty
            w_avoid: Weight for collision avoidance reward (paper Section 4)
            collision_C: Collision penalty constant C ~1 (paper Section 4)
        """
        self.num_drones = num_drones
        self.collision_dist = collision_dist
        self.w_form = w_form
        self.w_nav = w_nav
        self.w_dist = w_dist
        self.w_avoid = w_avoid
        self.collision_C = collision_C
        
        # RANDOMIZED Initial positions for generalization
        # Range: x,y ∈ [-1.5, 1.5], z ∈ [0.1, 0.5]
        # This ensures diverse starting configurations during training
        initial_xyzs = np.zeros((num_drones, 3))
        for i in range(num_drones):
            initial_xyzs[i] = [
                np.random.uniform(-1.5, 1.5),  # Random X
                np.random.uniform(-1.5, 1.5),  # Random Y
                np.random.uniform(0.1, 0.5)     # Random Z (above ground)
            ]
        
        # Create the underlying environment
        # ActionType.VEL accepts velocity commands and uses internal PID to compute RPMs
        # This means we don't need our own PID controller
        self._env = MultiHoverAviary(
            drone_model=DroneModel.CF2X,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs_type,
            act=act_type,
        )
        
        # Store for computing deltas
        self._prev_distances = None
        
        # Visual markers for rendering
        self._visual_marker_ids = []  # Store IDs of visual markers for cleanup
        
        # Set up spaces for MAPPO
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Set up observation, share_observation, and action spaces for MAPPO."""
        # Get base observation space from env
        base_obs_space = self._env.observation_space
        
        # Per-agent observation dimension
        # New KIN obs: own_state(6) + rel_target(3) + neighbors(6*(N-1)) = 21 for 3 drones
        if hasattr(base_obs_space, 'shape'):
            self.obs_dim = base_obs_space.shape[1] if len(base_obs_space.shape) > 1 else base_obs_space.shape[0]
        else:
            # For 3 drones: 6 + 3 + 6*2 = 21
            self.obs_dim = 6 + 3 + 6 * (self.num_drones - 1)
        
        # Share observation: positions (3) + velocities (3) + target_pos (3) for all agents
        # Gives the centralized critic full information about where drones are and should go
        self.share_obs_dim_per_agent = 9  # pos(3) + vel(3) + target(3)
        self.share_obs_dim = self.num_drones * self.share_obs_dim_per_agent
        
        # Create spaces (lists for MAPPO compatibility)
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_drones)
        ]
        
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_drones)
        ]
        
        # Action space: 4D velocity command [vx, vy, vz, throttle_scale]
        # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 3 (Action & controller)
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for _ in range(self.num_drones)
        ]
        
        # For env_wrappers compatibility
        self.n = self.num_drones
    
    def seed(self, seed: int = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def _get_drone_states(self) -> np.ndarray:
        """
        Get states of all drones.
        
        Returns:
            Array of shape (num_drones, state_dim) containing drone states
        """
        states = np.array([
            self._env._getDroneStateVector(i) for i in range(self.num_drones)
        ])
        return states
    
    def _extract_positions_velocities(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract positions and velocities from drone states.
        
        Args:
            states: Drone states array (num_drones, state_dim)
            
        Returns:
            positions: (num_drones, 3)
            velocities: (num_drones, 3)
        """
        positions = states[:, 0:3]
        velocities = states[:, 10:13]
        return positions, velocities
    
    def _compute_share_obs(self, states: np.ndarray) -> np.ndarray:
        """
        Compute shared observation for centralized critic.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 2, 7
        - share_obs: concatenated positions and velocities of all agents
        - Tiled per agent (each agent sees the same global state)
        
        Args:
            states: Drone states array (num_drones, state_dim)
            
        Returns:
            share_obs: Array of shape (num_drones, share_obs_dim)
        """
        positions, velocities = self._extract_positions_velocities(states)
        
        # Get target positions from the underlying env
        target_pos = self._env.TARGET_POS  # (num_drones, 3)
        
        # Concatenate pos + vel + target for each agent, then flatten
        agent_share_info = np.concatenate([positions, velocities, target_pos], axis=1)  # (num_drones, 9)
        share_obs_flat = agent_share_info.flatten()  # (num_drones * 9,)
        
        # Tile for each agent (each agent gets the same share_obs)
        share_obs = np.tile(share_obs_flat, (self.num_drones, 1))  # (num_drones, share_obs_dim)
        
        return share_obs.astype(np.float32)
    
    def _compute_per_agent_obs(self, raw_obs: np.ndarray) -> List[np.ndarray]:
        """
        Convert raw observations to per-agent list format.
        
        Args:
            raw_obs: Raw observations from env (num_drones, obs_dim)
            
        Returns:
            List of observations, one per agent
        """
        return [raw_obs[i].astype(np.float32) for i in range(self.num_drones)]
    
    def _compute_reward(
        self, 
        states: np.ndarray,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Compute reward for navigation with formation alignment.
        
        Total reward (shared across all agents):
            r = w_form * r_form + w_nav * r_nav + w_dist * r_dist + w_avoid * r_avoid + r_reached
        
        Components:
        - Formation error via Procrustes/SVD alignment (downweighted):
            r_form = -E / (G + eps)
        - Navigation reward — PRIMARY signal (delta distance improvement):
            r_nav = sum_i (d_prev_i - d_curr_i)
        - Distance penalty — persistent penalty for being far from targets:
            r_dist = -(1/N) * sum_i dist_i
        - Collision avoidance penalty:
            r_avoid = -C * num_collisions
        - Reaching bonus: +1.0 per drone within 5cm of its target
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 4
        """
        positions, _ = self._extract_positions_velocities(states)
        target_pos = self._env.TARGET_POS  # (num_drones, 3)
        
        # --- 1. Formation reward (Procrustes/SVD) — downweighted ---
        E, G = compute_formation_error(positions, target_pos)
        r_form = -E / (G + 1e-8)
        
        # --- 2. Navigation reward — PRIMARY signal ---
        # r_nav = sum_i (d_prev_i - d_curr_i)
        r_nav = 0.0
        current_distances = np.zeros(self.num_drones)
        for i in range(self.num_drones):
            dist = np.linalg.norm(target_pos[i] - positions[i])
            current_distances[i] = dist
            r_nav += (self._prev_distances[i] - dist)
            self._prev_distances[i] = dist
        
        # --- 3. Distance penalty — persistent signal ---
        # Penalizes current distance to targets (provides gradient even when stationary)
        r_dist = -np.mean(current_distances)
        
        # --- 4. Collision avoidance penalty ---
        num_collisions = 0
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                inter_dist = np.linalg.norm(positions[i] - positions[j])
                if inter_dist < self.collision_dist:
                    num_collisions += 1
        r_avoid = -self.collision_C * num_collisions
        
        # --- 5. Reaching bonus ---
        # +1.0 per drone within 5cm of its target
        r_reached = sum(1.0 for d in current_distances if d < 0.50)
        
        # --- Total reward (shared across all agents) ---
        total_reward = (self.w_form * r_form 
                       + self.w_nav * r_nav 
                       + self.w_dist * r_dist 
                       + self.w_avoid * r_avoid 
                       + r_reached)
        
        # All agents receive the same shared reward (paper Section 4)
        rewards = [total_reward] * self.num_drones
        
        # Compute per-drone distances for logging
        per_drone_dists = current_distances.tolist()
        
        reward_info = {
            'r_form': r_form,
            'r_nav': r_nav,
            'r_dist': r_dist,
            'r_avoid': r_avoid,
            'r_reached': r_reached,
            'formation_error': E,
            'normalization_G': G,
            'num_collisions': num_collisions,
            'total_reward': total_reward,
        }
        reward_infos = []
        for i in range(self.num_drones):
            info_i = reward_info.copy()
            info_i['dist_to_target'] = per_drone_dists[i]
            info_i['current_pos'] = positions[i].tolist() if hasattr(positions[i], 'tolist') else list(positions[i])
            info_i['target_pos'] = target_pos[i].tolist() if hasattr(target_pos[i], 'tolist') else list(target_pos[i])
            reward_infos.append(info_i)
        
        return rewards, reward_infos
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset the environment.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 6
        
        Returns:
            obs_n: Per-agent observations as numpy array (num_drones, obs_dim)
            share_obs: Shared observation for centralized critic (num_drones, share_obs_dim)
            available_actions: Available actions mask (all available for continuous control)
        """
        raw_obs, info = self._env.reset()
        
        # Get states for share_obs computation
        states = self._get_drone_states()
        positions, _ = self._extract_positions_velocities(states)
        
        # Store initial positions and targets for logging
        self._episode_initial_positions = positions.copy()
        self._episode_target_positions = self._env.TARGET_POS.copy()
        self._episode_step_count = 0
        self._episode_cumulative_reward = 0.0
        self._episode_terminated_early = False
        
        # Initialize per-drone distances to TARGET_POS for shaped reward
        self._prev_distances = np.array([
            np.linalg.norm(self._env.TARGET_POS[i] - positions[i])
            for i in range(self.num_drones)
        ])
        self._episode_initial_distances = self._prev_distances.copy()
        
        # Compute observations - return as numpy arrays instead of lists
        obs_n = raw_obs.astype(np.float32)  # Already (num_drones, obs_dim)
        share_obs = self._compute_share_obs(states)  # (num_drones, share_obs_dim)
        
        # All actions available for continuous control
        available_actions = np.ones((self.num_drones, 4), dtype=np.float32)
        
        return obs_n, share_obs, available_actions
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[bool], List[Dict], np.ndarray]:
        """
        Take a step in the environment.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 6
        
        Args:
            actions: Actions for all agents (num_drones, action_dim)
                    For VEL action type: [vx, vy, vz, throttle_scale] in [-1, 1]
        
        Returns:
            obs_n: Per-agent observations as numpy array (num_drones, obs_dim)
            share_obs: Shared observation (num_drones, share_obs_dim)
            rewards_n: List of [reward] per agent
            dones_n: List of done flags per agent
            infos_n: List of info dicts per agent
            available_actions: Available actions mask (all available for continuous control)
        """
        # Ensure actions are numpy array with correct shape
        if isinstance(actions, list):
            actions = np.array(actions)
        
        if len(actions.shape) == 1:
            actions = actions.reshape(self.num_drones, -1)
        
        # Take step in environment
        # MultiHoverAviary with ActionType.VEL handles PID internally
        raw_obs, base_reward, terminated, truncated, info = self._env.step(actions)
        
        # Combine terminated and truncated for done
        done = terminated or truncated
        
        # Get states for reward computation and share_obs
        states = self._get_drone_states()
        
        # Compute reward per paper Section 4 (formation + navigation + collision)
        rewards_list, reward_infos = self._compute_reward(states)
        
        # Compute observations - return as numpy arrays
        obs_n = raw_obs.astype(np.float32)  # Already (num_drones, obs_dim)
        share_obs = self._compute_share_obs(states)  # (num_drones, share_obs_dim)
        
        # MAPPO expects list format for rewards and dones
        rewards_n = [[r] for r in rewards_list]  # Per-drone individual rewards
        dones_n = [done for _ in range(self.num_drones)]
        
        # Track episode progress
        self._episode_step_count += 1
        self._episode_cumulative_reward += reward_infos[0].get('total_reward', rewards_list[0])
        if terminated:
            self._episode_terminated_early = True
        
        # Build info dicts
        infos_n = []
        positions, velocities = self._extract_positions_velocities(states)
        for i in range(self.num_drones):
            agent_info = {
                'individual_reward': rewards_list[i],
                'current_position': positions[i].tolist(),
                'current_velocity': velocities[i].tolist(),
                **reward_infos[i]
            }
            # Add episode summary info on done (accessible by runner before auto-reset)
            if done:
                agent_info['episode_summary'] = {
                    'initial_position': self._episode_initial_positions[i].tolist(),
                    'final_position': positions[i].tolist(),
                    'target_position': self._episode_target_positions[i].tolist(),
                    'initial_distance': float(self._episode_initial_distances[i]),
                    'final_distance': float(reward_infos[i]['dist_to_target']),
                    'distance_improvement': float(self._episode_initial_distances[i] - reward_infos[i]['dist_to_target']),
                    'reached_target': reward_infos[i]['dist_to_target'] < 0.50,
                    'direction_traveled': (positions[i] - self._episode_initial_positions[i]).tolist(),
                    'direction_to_target': (self._episode_target_positions[i] - self._episode_initial_positions[i]).tolist(),
                }
                agent_info['episode_steps'] = self._episode_step_count
                agent_info['episode_cumulative_reward'] = self._episode_cumulative_reward
                agent_info['episode_terminated_early'] = self._episode_terminated_early
            infos_n.append(agent_info)
        
        # All actions available for continuous control
        available_actions = np.ones((self.num_drones, 4), dtype=np.float32)
        
        return obs_n, share_obs, rewards_n, dones_n, infos_n, available_actions
    
    def draw_position_markers(self):
        """Draw visual markers for initial and target positions in PyBullet GUI."""
        try:
            import pybullet as p
            
            # Remove old markers
            for marker_id in self._visual_marker_ids:
                try:
                    p.removeBody(marker_id)
                except:
                    pass
            self._visual_marker_ids.clear()
            
            # Draw markers for each drone
            for i in range(self.num_drones):
                # Green sphere for initial position
                initial_pos = self._episode_initial_positions[i]
                initial_visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.1,
                    rgbaColor=[0, 1, 0, 0.5]  # Green, semi-transparent
                )
                initial_marker = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=initial_visual,
                    basePosition=initial_pos
                )
                self._visual_marker_ids.append(initial_marker)
                
                # Red sphere for target position
                target_pos = self._episode_target_positions[i]
                target_visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.15,
                    rgbaColor=[1, 0, 0, 0.6]  # Red, semi-transparent
                )
                target_marker = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=target_visual,
                    basePosition=target_pos
                )
                self._visual_marker_ids.append(target_marker)
                
                # Draw a line from initial to target position
                p.addUserDebugLine(
                    lineFromXYZ=initial_pos,
                    lineToXYZ=target_pos,
                    lineColorRGB=[0.5, 0.5, 0.5],
                    lineWidth=2,
                    lifeTime=0  # Permanent until removed
                )
                
        except Exception as e:
            # Silently fail if PyBullet not available or GUI not active
            pass
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        return self._env.render()
    
    def close(self):
        """Close the environment."""
        # Clean up visual markers
        try:
            import pybullet as p
            for marker_id in self._visual_marker_ids:
                try:
                    p.removeBody(marker_id)
                except:
                    pass
        except:
            pass
        self._env.close()


def make_pybullet_drone_env(all_args):
    """
    Factory function to create PyBulletDroneWrapper environment.
    
    Compatible with MAPPO's env creation pattern.
    
    Args:
        all_args: Namespace with environment arguments
        
    Returns:
        PyBulletDroneWrapper environment
    """
    num_drones = getattr(all_args, 'num_drones', 3)
    gui = getattr(all_args, 'render', False)
    
    env = PyBulletDroneWrapper(
        num_drones=num_drones,
        gui=gui,
    )
    
    return env
