# Training Strategy for MARL-Based Formation Control (Aligned with SSRN MA-LSTM-PPO)

This document specifies the **correct training methodology for formation control** using MARL,
aligned with the SSRN MA-LSTM-PPO paper and standard drone MARL practice.

This replaces the incorrect approach of:
> Randomly assigning completely independent initial and target positions to each drone in every episode.

Formation control is a **structured cooperative control problem**, not independent navigation.

---

# 1. Core Task Definition

We are training a policy to:

- Maintain a **fixed relative formation geometry**
- Navigate toward a **common swarm target**
- Avoid collisions
- Remain decentralized at execution time

This is NOT trajectory imitation (e.g., circle-following).
This is NOT independent drone navigation.

The task is:

> Maintain relative geometric structure while moving toward arbitrary goals.

---

# 2. What Should NOT Be Done

❌ Random independent start position for each drone  
❌ Random independent target per drone  
❌ Random formation geometry per episode (unless explicitly studying adaptive formations)  
❌ Hard-coded circle or line trajectories during training  

These approaches break the core structure of formation control.

---

# 3. Correct Training Structure

## 3.1 Fixed Formation Geometry

Define a base formation template.

Example (3 drones):

Drone 1: (0, 0, 0)  
Drone 2: (d, 0, 0)  
Drone 3: (d/2, √3 d / 2, 0)

This defines desired **relative positions**, not absolute positions.

This geometry remains constant during training.

---

## 3.2 Episode Reset Procedure

Each episode should:

### Step 1: Randomize Formation Center

Select a random formation center within arena bounds:

center ~ Uniform(arena_bounds)

### Step 2: Randomize Formation Orientation

Select random rotation:

θ ~ Uniform(0, 2π)

Rotate entire formation template by θ.

### Step 3: Add Small Perturbations

Add small Gaussian noise to each drone position:

position_i += N(0, σ_small)

Purpose:
- Teach stabilization
- Prevent memorization

### Step 4: Randomize Common Target Position

Select a single target for the entire swarm:

target ~ Uniform(arena_bounds)

All drones share this target.

---

# 4. Reward Structure (Aligned with SSRN)

Total reward per drone:

r = r_formation + w1 * r_navigation + w2 * r_collision

## 4.1 Formation Reward

- Based on rigid-body invariant alignment
- Penalizes shape distortion
- Independent of global translation and rotation

## 4.2 Navigation Reward

Encourages reduction in distance between swarm center and target.

## 4.3 Collision Penalty

Strong negative reward if inter-drone distance < safety threshold.

---

# 5. Why This Training Strategy Works

This structure ensures the policy learns:

- Translation invariance
- Rotation invariance
- Direction invariance
- Cooperative stabilization
- Collision avoidance under perturbation

The trajectory emerges naturally from reward optimization.


# 6. Summary

Correct formation control training requires:

- Fixed relative geometry
- Random global translation
- Random global rotation
- Small perturbations
- Shared target per episode
- Structured reward

Training independent start/goal per drone breaks formation assumptions.

---

# Final Principle

Formation control training must preserve structure while introducing variability.

Do NOT randomize structure.
Randomize configuration and context.

This ensures generalizable, decentralized formation behavior suitable for real Crazyflie deployment.
