"""
Unit tests for MA-LSTM-PPO integration.

These smoke tests verify that:
1. Formation utilities work correctly
2. PID controller functions properly
3. MA-LSTM policy has correct API and shapes
4. Environment wrapper produces correct observation shapes

Reference: docs/MA-LSTM-PPO-paper-summary.md Section 8 (Minimal acceptance tests)

Run with: pytest tests/test_malstm_integration.py -v
"""

import pytest
import numpy as np
import torch
from gym import spaces


def _pybullet_available():
    """Check if PyBullet is available."""
    try:
        import pybullet
        return True
    except ImportError:
        return False


class TestFormationUtils:
    """Tests for onpolicy/utils/formation.py"""
    
    def test_procrustes_alignment_identity(self):
        """Test Procrustes alignment with identical point sets."""
        from onpolicy.utils.formation import procrustes_alignment
        
        # Create random points
        points = np.random.randn(5, 3)
        
        # Align with itself
        R, aligned, error = procrustes_alignment(points, points)
        
        # Should have zero error (within numerical precision)
        assert error < 1e-10, f"Expected near-zero error, got {error}"
        
        # Rotation should be identity
        assert np.allclose(R, np.eye(3), atol=1e-6), "Rotation should be identity"
    
    def test_procrustes_alignment_rotation(self):
        """Test Procrustes alignment with rotated point set."""
        from onpolicy.utils.formation import procrustes_alignment
        
        # Create points
        source = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ], dtype=np.float64)
        
        # Rotate by 90 degrees around z-axis
        R_true = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        
        target = source @ R_true.T
        
        # Recover rotation
        R, aligned, error = procrustes_alignment(source, target)
        
        # Should recover the rotation
        assert np.allclose(R, R_true, atol=1e-6), f"Rotation mismatch: {R} vs {R_true}"
        assert error < 1e-10, f"Expected near-zero error, got {error}"
    
    def test_formation_error_perfect(self):
        """Test formation error with perfect alignment."""
        from onpolicy.utils.formation import compute_formation_error
        
        formation = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0.5, 0.866, 1],  # Equilateral triangle
        ])
        
        E, G = compute_formation_error(formation, formation)
        
        assert E < 1e-10, f"Expected zero error, got {E}"
        assert G > 0, "Normalization factor should be positive"
    
    def test_formation_error_translated(self):
        """Test formation error is translation-invariant."""
        from onpolicy.utils.formation import compute_formation_error
        
        formation = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0.5, 0.866, 1],
        ])
        
        # Translate positions
        translated = formation + np.array([10, 20, 30])
        
        E, G = compute_formation_error(translated, formation)
        
        # Should still have zero error (formation shape is same)
        assert E < 1e-10, f"Expected zero error after translation, got {E}"
    
    def test_normalization_factor(self):
        """Test normalization factor computation."""
        from onpolicy.utils.formation import compute_normalization_factor
        
        # Unit square formation
        formation = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ])
        
        G = compute_normalization_factor(formation)
        
        # Max distance is diagonal = sqrt(2), so G = 2
        expected_G = 2.0
        assert np.isclose(G, expected_G, atol=1e-6), f"Expected G={expected_G}, got {G}"
    
    def test_generate_formation_shapes(self):
        """Test formation generation for different shapes."""
        from onpolicy.utils.formation import generate_formation
        
        for formation_type in ['triangle', 'line', 'circle', 'grid']:
            formation = generate_formation(formation_type, num_agents=4)
            
            assert formation.shape == (4, 3), f"Wrong shape for {formation_type}"
            assert not np.isnan(formation).any(), f"NaN values in {formation_type}"


class TestPIDController:
    """Tests for onpolicy/controllers/pid_controller.py"""
    
    def test_pid_zero_error(self):
        """Test PID controller with zero velocity error."""
        from onpolicy.controllers.pid_controller import PIDController
        
        controller = PIDController(dt=1/30)
        
        v_des = np.array([1.0, 0.0, 0.5])
        v_current = np.array([1.0, 0.0, 0.5])  # No error
        
        control = controller.compute(v_des, v_current)
        
        # With zero error, output should be near zero
        assert np.allclose(control, 0, atol=1e-6), f"Expected zero control, got {control}"
    
    def test_pid_positive_error(self):
        """Test PID controller with positive velocity error."""
        from onpolicy.controllers.pid_controller import PIDController
        
        controller = PIDController(dt=1/30)
        
        v_des = np.array([1.0, 0.0, 0.0])
        v_current = np.array([0.0, 0.0, 0.0])
        
        control = controller.compute(v_des, v_current)
        
        # Should have positive x-component
        assert control[0] > 0, f"Expected positive x control, got {control[0]}"
    
    def test_pid_reset(self):
        """Test PID controller reset functionality."""
        from onpolicy.controllers.pid_controller import PIDController
        
        controller = PIDController(dt=1/30)
        
        # Build up some integral
        for _ in range(10):
            controller.compute(np.array([1, 0, 0]), np.array([0, 0, 0]))
        
        # Reset
        controller.reset()
        
        # Integral should be zero
        assert np.allclose(controller.integral, 0), "Integral not reset"
    
    def test_velocity_to_motor_controller(self):
        """Test velocity to motor conversion."""
        from onpolicy.controllers.pid_controller import VelocityToMotorController
        
        controller = VelocityToMotorController()
        
        v_des = np.array([0, 0, 0, 0])  # Hover
        state = {'velocity': np.zeros(3)}
        
        rpms = controller.compute(v_des, state)
        
        assert rpms.shape == (4,), f"Wrong RPM shape: {rpms.shape}"
        assert (rpms > 0).all(), "RPMs should be positive for hover"


class TestMALSTMPolicy:
    """Tests for onpolicy/models/ma_lstm_policy.py"""
    
    @pytest.fixture
    def policy_setup(self):
        """Create a mock args and spaces for policy initialization."""
        class Args:
            hidden_size = 64
            use_orthogonal = True
            recurrent_N = 1
            lr = 5e-4
            critic_lr = 5e-4
            opti_eps = 1e-5
            weight_decay = 0
        
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)  # 3 agents * 6
        act_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        return Args(), obs_space, share_obs_space, act_space
    
    def test_policy_initialization(self, policy_setup):
        """Test MA-LSTM policy initialization."""
        from onpolicy.models.ma_lstm_policy import MA_LSTM_Policy
        
        args, obs_space, share_obs_space, act_space = policy_setup
        
        policy = MA_LSTM_Policy(
            args, obs_space, share_obs_space, act_space,
            device=torch.device("cpu")
        )
        
        assert policy is not None
        assert hasattr(policy, 'actor')
        assert hasattr(policy, 'critic')
    
    def test_policy_get_actions_shape(self, policy_setup):
        """
        Test that get_actions returns correct shapes.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 8
        - policy.act() with random obs returns actions shaped (batch, action_dim)
        """
        from onpolicy.models.ma_lstm_policy import MA_LSTM_Policy
        
        args, obs_space, share_obs_space, act_space = policy_setup
        
        policy = MA_LSTM_Policy(
            args, obs_space, share_obs_space, act_space,
            device=torch.device("cpu")
        )
        
        batch_size = 4
        obs = np.random.randn(batch_size, 12).astype(np.float32)
        share_obs = np.random.randn(batch_size, 18).astype(np.float32)
        
        # RNN states: (2, num_layers, batch, hidden) for LSTM (h and c)
        rnn_states_actor = np.zeros((2, 1, batch_size, 64), dtype=np.float32)
        rnn_states_critic = np.zeros((2, 1, batch_size, 64), dtype=np.float32)
        masks = np.ones((batch_size, 1), dtype=np.float32)
        
        values, actions, log_probs, rnn_actor_out, rnn_critic_out = policy.get_actions(
            share_obs, obs, rnn_states_actor, rnn_states_critic, masks
        )
        
        # Check shapes
        assert values.shape == (batch_size, 1), f"Wrong values shape: {values.shape}"
        assert actions.shape == (batch_size, 4), f"Wrong actions shape: {actions.shape}"
        assert log_probs.shape == (batch_size, 1), f"Wrong log_probs shape: {log_probs.shape}"
    
    def test_policy_evaluate_actions(self, policy_setup):
        """
        Test evaluate_actions returns correct shapes.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 2 (I/O contract)
        """
        from onpolicy.models.ma_lstm_policy import MA_LSTM_Policy
        
        args, obs_space, share_obs_space, act_space = policy_setup
        
        policy = MA_LSTM_Policy(
            args, obs_space, share_obs_space, act_space,
            device=torch.device("cpu")
        )
        
        batch_size = 4
        obs = np.random.randn(batch_size, 12).astype(np.float32)
        share_obs = np.random.randn(batch_size, 18).astype(np.float32)
        actions = np.random.randn(batch_size, 4).astype(np.float32)
        
        rnn_states_actor = np.zeros((2, 1, batch_size, 64), dtype=np.float32)
        rnn_states_critic = np.zeros((2, 1, batch_size, 64), dtype=np.float32)
        masks = np.ones((batch_size, 1), dtype=np.float32)
        
        values, log_probs, entropy = policy.evaluate_actions(
            share_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks
        )
        
        assert values.shape == (batch_size, 1), f"Wrong values shape: {values.shape}"
        assert log_probs.shape == (batch_size, 1), f"Wrong log_probs shape: {log_probs.shape}"
        assert isinstance(entropy.item(), float), "Entropy should be scalar"
    
    def test_policy_deterministic_vs_stochastic(self, policy_setup):
        """Test that deterministic mode produces same actions."""
        from onpolicy.models.ma_lstm_policy import MA_LSTM_Policy
        
        args, obs_space, share_obs_space, act_space = policy_setup
        
        policy = MA_LSTM_Policy(
            args, obs_space, share_obs_space, act_space,
            device=torch.device("cpu")
        )
        
        batch_size = 4
        obs = np.random.randn(batch_size, 12).astype(np.float32)
        rnn_states = np.zeros((2, 1, batch_size, 64), dtype=np.float32)
        masks = np.ones((batch_size, 1), dtype=np.float32)
        
        # Deterministic should give same result
        actions1, _ = policy.act(obs, rnn_states, masks, deterministic=True)
        actions2, _ = policy.act(obs, rnn_states, masks, deterministic=True)
        
        assert torch.allclose(actions1, actions2), "Deterministic actions should be identical"


class TestPyBulletWrapper:
    """Tests for onpolicy/envs/pybullet_drone_env.py"""
    
    @pytest.mark.skipif(
        not _pybullet_available(),
        reason="PyBullet not available"
    )
    def test_env_reset_shapes(self):
        """
        Test environment reset returns correct shapes.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 8
        - env.reset() returns shapes: obs.shape == (n_agents, obs_dim)
        """
        from onpolicy.envs.pybullet_drone_env import PyBulletDroneWrapper
        
        num_drones = 3
        env = PyBulletDroneWrapper(num_drones=num_drones, gui=False)
        
        try:
            obs, share_obs = env.reset()
            
            assert len(obs) == num_drones, f"Expected {num_drones} obs, got {len(obs)}"
            assert share_obs.shape[0] == num_drones, f"Wrong share_obs shape: {share_obs.shape}"
            
            # Each agent should have same obs dim
            obs_dims = [o.shape[0] for o in obs]
            assert len(set(obs_dims)) == 1, "All agents should have same obs dim"
        finally:
            env.close()
    
    @pytest.mark.skipif(
        not _pybullet_available(),
        reason="PyBullet not available"
    )
    def test_env_step_no_exception(self):
        """
        Test environment step executes without exception.
        
        Reference: docs/MA-LSTM-PPO-paper-summary.md Section 8
        - A single env.step(actions) executes with no exception
        """
        from onpolicy.envs.pybullet_drone_env import PyBulletDroneWrapper
        
        num_drones = 3
        env = PyBulletDroneWrapper(num_drones=num_drones, gui=False)
        
        try:
            obs, share_obs = env.reset()
            
            # Random actions
            actions = np.random.uniform(-1, 1, (num_drones, 4))
            
            # Step should not raise
            obs_new, share_obs_new, rewards, dones, infos = env.step(actions)
            
            # Basic sanity checks
            assert len(obs_new) == num_drones
            assert len(rewards) == num_drones
            assert len(dones) == num_drones
            assert len(infos) == num_drones
        finally:
            env.close()
    
    @pytest.mark.skipif(
        not _pybullet_available(),
        reason="PyBullet not available"
    )
    def test_env_spaces(self):
        """Test environment has correct space attributes."""
        from onpolicy.envs.pybullet_drone_env import PyBulletDroneWrapper
        
        num_drones = 3
        env = PyBulletDroneWrapper(num_drones=num_drones, gui=False)
        
        try:
            assert len(env.observation_space) == num_drones
            assert len(env.share_observation_space) == num_drones
            assert len(env.action_space) == num_drones
            assert env.n == num_drones
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
