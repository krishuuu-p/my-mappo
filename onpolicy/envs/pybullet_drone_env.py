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
from onpolicy.utils.formation import compute_formation_error, compute_normalization_factor


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
        ctrl_freq: int = 30,
        episode_len_sec: float = 8.0,
        target_formation: Optional[np.ndarray] = None,
        use_formation_reward: bool = True,
        w_nav: float = 0.1,
        w_avoid: float = 1.0,
        collision_dist: float = 0.1,
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
            ctrl_freq: Control frequency
            episode_len_sec: Episode length in seconds
            target_formation: Target formation shape (num_drones, 3) or None for default
            use_formation_reward: Whether to use formation-based reward
            w_nav: Weight for navigation reward (paper default)
            w_avoid: Weight for collision avoidance penalty
            collision_dist: Distance threshold for collision detection
            
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 5 (hyperparameters)
        """
        self.num_drones = num_drones
        self.use_formation_reward = use_formation_reward
        self.w_nav = w_nav
        self.w_avoid = w_avoid
        self.collision_dist = collision_dist
        
        # Default triangular formation if not specified
        if target_formation is None:
            # Create a simple triangular formation
            angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)
            radius = 0.5
            self.target_formation = np.stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.ones(num_drones)  # All at height 1.0
            ], axis=1)
        else:
            self.target_formation = target_formation
        
        # Initial positions: spread out horizontally
        initial_xyzs = np.zeros((num_drones, 3))
        for i in range(num_drones):
            initial_xyzs[i] = [0.3 * (i - num_drones/2), 0, 0.1]
        
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
        
        # Share observation: positions (3) + velocities (3) for all agents, concatenated
        # Reference: docs/MA-LSTM-PPO-paper-summary.md Section 2 (Centralized Critic input)
        self.share_obs_dim_per_agent = 6  # pos(3) + vel(3)
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
        
        # Concatenate pos and vel for each agent, then flatten
        agent_share_info = np.concatenate([positions, velocities], axis=1)  # (num_drones, 6)
        share_obs_flat = agent_share_info.flatten()  # (num_drones * 6,)
        
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
    
    def _compute_formation_reward(
        self, 
        positions: np.ndarray,
        prev_positions: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute formation-based reward as per MA-LSTM-PPO paper.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 4 (Reward)
        
        r = r_form + w_nav * r_nav + w_avoid * r_avoid
        
        Args:
            positions: Current drone positions (num_drones, 3)
            prev_positions: Previous positions for navigation reward
            
        Returns:
            total_reward: Scalar reward
            reward_info: Dict with reward components
        """
        # Formation error using Procrustes alignment
        E, G = compute_formation_error(positions, self.target_formation)
        r_form = -E / (G + 1e-8)
        
        # Navigation reward: reduction in distance to target formation center
        r_nav = 0.0
        if prev_positions is not None:
            target_center = np.mean(self.target_formation, axis=0)
            prev_center = np.mean(prev_positions, axis=0)
            curr_center = np.mean(positions, axis=0)
            prev_dist = np.linalg.norm(prev_center - target_center)
            curr_dist = np.linalg.norm(curr_center - target_center)
            r_nav = prev_dist - curr_dist
        
        # Collision avoidance penalty
        num_collisions = 0
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.collision_dist:
                    num_collisions += 1
        r_avoid = -num_collisions
        
        # Total reward
        total_reward = r_form + self.w_nav * r_nav + self.w_avoid * r_avoid
        
        reward_info = {
            'r_form': r_form,
            'r_nav': r_nav,
            'r_avoid': r_avoid,
            'formation_error': E,
        }
        
        return total_reward, reward_info
    
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
        
        # Store for computing navigation reward
        self._prev_positions = positions.copy()
        
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
        positions, _ = self._extract_positions_velocities(states)
        
        # Compute formation reward if enabled
        if self.use_formation_reward:
            reward, reward_info = self._compute_formation_reward(positions, self._prev_positions)
        else:
            reward = base_reward
            reward_info = {}
        
        # Update previous positions
        self._prev_positions = positions.copy()
        
        # Compute observations - return as numpy arrays
        obs_n = raw_obs.astype(np.float32)  # Already (num_drones, obs_dim)
        share_obs = self._compute_share_obs(states)  # (num_drones, share_obs_dim)
        
        # MAPPO expects list format for rewards and dones
        rewards_n = [[reward / self.num_drones] for _ in range(self.num_drones)]  # Shared reward
        dones_n = [done for _ in range(self.num_drones)]
        
        # Build info dicts
        infos_n = []
        for i in range(self.num_drones):
            agent_info = {
                'individual_reward': reward / self.num_drones,
                **reward_info
            }
            infos_n.append(agent_info)
        
        # All actions available for continuous control
        available_actions = np.ones((self.num_drones, 4), dtype=np.float32)
        
        return obs_n, share_obs, rewards_n, dones_n, infos_n, available_actions
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        return self._env.render()
    
    def close(self):
        """Close the environment."""
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
    use_formation_reward = getattr(all_args, 'use_formation_reward', True)
    
    env = PyBulletDroneWrapper(
        num_drones=num_drones,
        gui=gui,
        use_formation_reward=use_formation_reward,
    )
    
    return env
