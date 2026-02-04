# MA-LSTM-PPO — concise machine-readable summary (for code integration)

Source: local PDF `/mnt/data/ssrn-5286784.pdf` (MA-LSTM-PPO)
Purpose: implement CTDE multi-agent PPO variant with LSTM actor+critic and PyBullet drone sim.

---

## 1) High level algorithm (CTDE, PPO-style)
- Multi-agent, centralized training & decentralized execution (CTDE).
- Use PPO clipping objective with GAE for advantage estimation.
- Policies: recurrent LSTM actor (per-agent). Critic: centralized value network (may be LSTM).
- Training loop (high level):
  1. Collect rollouts for `T` timesteps across `N_envs` and `N_agents` using current policy.
  2. Compute rewards, values, GAE advantages.
  3. Flatten buffer and perform `K` PPO epochs of minibatch updates using clip `epsilon`.
  4. Update policy/critic parameters, repeat.

Pseudocode:
```
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
```

---

## 2) Model architecture (paper-specified)
- **Actor (per-agent, recurrent)**:
  - Input: agent observation vector `o_i` (obs_dim).
  - Dense → ReLU → LSTM (hidden H) → Dense → action mean `μ` (action_dim).
  - Learnable `log_std` parameter (diagonal Gaussian).
  - Output: continuous `v_des = [v_x, v_y, v_z, thrust]` or d-dim desired-velocity.
  - Optional `tanh` to bound outputs scaled to environment ranges.

- **Centralized Critic**:
  - Input: `share_obs` (global state: concatenated positions, velocities of all agents).
  - Dense → ReLU → (optional LSTM) → Dense → scalar value `V(s)`.

- **I/O contract** (must match MAPPO runner):
  ```python
  def act(self, obs, share_obs=None, rnn_states=None, deterministic=False):
      # obs: tensor (batch, obs_dim)
      # share_obs: tensor (batch, share_dim) or tiled per-agent
      # returns: value, action, action_log_probs, rnn_states_out

  def evaluate_actions(self, obs, share_obs, actions, rnn_states=None):
      # returns: values, log_probs, dist_entropy
  ```

---

## 3) Action & controller
- Policy outputs `v_des` (desired linear velocity + throttle).  
- Use a **PID controller** to convert `v_des` → motor commands (rotor thrusts) if the underlying PyBullet drone env expects motor-level commands. If `gym-pybullet-drones` already accepts velocity commands, send `v_des` directly.

---

## 4) Reward (per paper)
- Let `positions = {p_i}`, `target_positions = {t_i}`.
- **Formation error** `E` computed by best-fitting rigid transform (Procrustes / Umeyama / SVD).
  - Center both sets: `P = positions - mean(positions)`, `T = targets - mean(targets)`.
  - Compute rotation `R` via SVD of `P^T T`. Error:
    ```
    E = (1/N) * sum_i || R p_i - t_i ||^2
    ```
- **Normalization G**: paper uses max pairwise distance squared of target formation or similar (use `G = max_pairwise_dist^2`).
- **Formation reward**:
  ```
  r_form = - E / (G + eps)
  ```
- **Navigation reward**: reduction in distance-to-goal:
  ```
  r_nav = sum_i (d_prev_i - d_curr_i)
  ```
- **Collision penalty**:
  ```
  r_avoid = - C * num_collisions  # C ~1
  ```
- **Total reward**:
  ```
  r = r_form + w_nav * r_nav + w_avoid * r_avoid
  ```

---

## 5) Key hyperparameters (paper defaults / suggested)
- `lr = 5e-4`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `ppo_clip = 0.2`
- `train_steps_per_update / batch_size ≈ 20000`
- `num_workers` (envs) = paper uses many; for local testing use 8; CI use 1–2.
- `episode_length ≈ 242` timesteps

---

## 6) Implementation mapping (files to create)
- `onpolicy/envs/pybullet_drone_env.py` — wrapper around `gym-pybullet-drones`:
  - `reset()` -> `(per_agent_obs, share_obs)`
  - `step(actions)` -> `(per_agent_obs, share_obs, rewards, dones, infos)`
  - compute formation error & rewards inside or via util

- `onpolicy/models/ma_lstm_policy.py` — LSTM actor + centralized critic:
  - implement `act()` and `evaluate_actions()` to MAPPO contract
  - apply tanh/rescale to actions if needed

- `onpolicy/utils/formation.py` — Procrustes / SVD routine to compute `E` and `G`

- `onpolicy/controllers/pid_controller.py` — map `v_des` -> motor commands if needed

- Small test files in `tests/` for smoke tests.

---

## 7) Notes & edge cases
- `share_obs` shape: some MAPPO forks expect shape `(batch, n_agents*share_dim)` while others tile per agent `(batch*n_agents, share_dim)`. Provide `_get_share_obs()` that returns both or a tiled version — keep consistent with your MAPPO code.
- If `gym-pybullet-drones` accepts velocity commands, skip PID conversion.
- Handle NaNs/div-by-zero (add small eps).
- Use device-agnostic PyTorch code (`.to(device)`).

---

## 8) Minimal acceptance tests
1. `env.reset()` returns shapes: `obs.shape == (n_agents, obs_dim)` or `(n_envs, n_agents, obs_dim)` as MAPPO expects.
2. `policy.act()` with random obs returns actions shaped `(batch, action_dim)` and a scalar value tensor.
3. A single `env.step(actions)` executes with no exception.
