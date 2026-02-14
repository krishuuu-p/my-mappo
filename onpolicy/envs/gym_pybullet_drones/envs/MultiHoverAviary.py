import numpy as np

from onpolicy.envs.gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from onpolicy.envs.gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def _build_formation_template(num_drones, spacing=0.5):
    """
    Build a fixed formation template centered at the origin.
    
    For 2 drones: line along x-axis.
    For 3 drones: equilateral triangle in the XY plane.
    For N drones: regular polygon in the XY plane.
    
    Parameters
    ----------
    num_drones : int
        Number of drones.
    spacing : float
        Inter-drone distance (meters).
        
    Returns
    -------
    ndarray
        (num_drones, 3) formation template centered at origin (z=0).
    """
    template = np.zeros((num_drones, 3))
    if num_drones == 1:
        return template
    elif num_drones == 2:
        template[0] = [-spacing / 2, 0, 0]
        template[1] = [ spacing / 2, 0, 0]
    elif num_drones == 3:
        # Equilateral triangle
        template[0] = [0, 0, 0]
        template[1] = [spacing, 0, 0]
        template[2] = [spacing / 2, spacing * np.sqrt(3) / 2, 0]
        # Center at origin
        centroid = template.mean(axis=0)
        template -= centroid
    else:
        # Regular polygon
        for i in range(num_drones):
            angle = 2 * np.pi * i / num_drones
            template[i] = [spacing * np.cos(angle), spacing * np.sin(angle), 0]
        centroid = template.mean(axis=0)
        template -= centroid
    return template


def _apply_formation(template, center, yaw, perturbation_std=0.0):
    """
    Place a formation template at a given center with a yaw rotation and optional perturbation.
    
    Parameters
    ----------
    template : ndarray
        (N, 3) formation template centered at origin.
    center : ndarray
        (3,) center position [x, y, z].
    yaw : float
        Rotation angle in radians (around z-axis).
    perturbation_std : float
        Std-dev of Gaussian noise added to each drone position (meters).
        
    Returns
    -------
    ndarray
        (N, 3) drone positions.
    """
    # 2D rotation matrix around z-axis
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1],
    ])
    positions = (R @ template.T).T + center
    if perturbation_std > 0:
        positions += np.random.normal(0, perturbation_std, positions.shape)
    # Clamp z to be above ground
    positions[:, 2] = np.clip(positions[:, 2], 0.05, None)
    return positions


class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: formation control with navigation."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 formation_spacing: float=0.5,
                 perturbation_std: float=0.05,
                 arena_xy_bound: float=1.5,
                 arena_z_range: tuple=(0.2, 1.0),
                 target_distance_range: tuple=(0.5, 1.5),
                 ):
        """Initialization of a multi-agent RL environment for formation control.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
            If None, positions are generated from the formation template.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        formation_spacing : float
            Inter-drone distance in the formation template (meters).
        perturbation_std : float
            Std-dev of Gaussian noise added to initial positions (meters).
        arena_xy_bound : float
            Max absolute x/y value for random formation center.
        arena_z_range : tuple
            (z_min, z_max) for random formation center height.
        target_distance_range : tuple
            (min, max) distance from initial center to target center.
        """
        self.EPISODE_LEN_SEC = 8
        
        # Store formation parameters
        self._formation_spacing = formation_spacing
        self._perturbation_std = perturbation_std
        self._arena_xy_bound = arena_xy_bound
        self._arena_z_range = arena_z_range
        self._target_distance_range = target_distance_range
        
        # Build fixed formation template (relative positions, centered at origin)
        self._formation_template = _build_formation_template(num_drones, formation_spacing)
        
        # Generate initial formation-based positions for first episode
        if initial_xyzs is None:
            initial_xyzs = self._generate_formation_positions()
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        # Override speed limit: 0.03 is too conservative for navigation tasks
        # 0.1 * MAX_SPEED gives ~0.83 m/s, enough to travel ~6.6m in 8s episode
        if hasattr(self, 'SPEED_LIMIT'):
            self.SPEED_LIMIT = 0.1 * self.MAX_SPEED_KMH * (1000/3600)
        
        # Generate target positions: same formation at a different random center
        self.TARGET_POS = self._generate_target_positions()

    ################################################################################

    def _generate_formation_positions(self):
        """
        Generate drone positions from the formation template with random
        center, random yaw orientation, and small perturbations.
        
        Returns
        -------
        ndarray
            (NUM_DRONES, 3) positions.
        """
        center = np.array([
            np.random.uniform(-self._arena_xy_bound, self._arena_xy_bound),
            np.random.uniform(-self._arena_xy_bound, self._arena_xy_bound),
            np.random.uniform(*self._arena_z_range),
        ])
        yaw = np.random.uniform(0, 2 * np.pi)
        positions = _apply_formation(
            self._formation_template, center, yaw,
            perturbation_std=self._perturbation_std,
        )
        self._current_init_center = center
        self._current_init_yaw = yaw
        return positions

    def _generate_target_positions(self):
        """
        Generate target positions: same formation template placed at a
        random target center (different from initial center).
        
        The formation orientation at the target may also be randomised
        so the policy learns rotation invariance.
        
        Returns
        -------
        ndarray
            (NUM_DRONES, 3) target positions.
        """
        # Random target center at a reachable distance from initial center
        dist = np.random.uniform(*self._target_distance_range)
        angle_xy = np.random.uniform(0, 2 * np.pi)
        z_target = np.random.uniform(*self._arena_z_range)
        
        target_center = np.array([
            self._current_init_center[0] + dist * np.cos(angle_xy),
            self._current_init_center[1] + dist * np.sin(angle_xy),
            z_target,
        ])
        # Clamp target center inside arena
        target_center[0] = np.clip(target_center[0], -self._arena_xy_bound - 1, self._arena_xy_bound + 1)
        target_center[1] = np.clip(target_center[1], -self._arena_xy_bound - 1, self._arena_xy_bound + 1)
        target_center[2] = np.clip(target_center[2], self._arena_z_range[0], self._arena_z_range[1])
        
        # Random yaw at target (may differ from initial yaw for rotation invariance)
        target_yaw = np.random.uniform(0, 2 * np.pi)
        
        target_positions = _apply_formation(
            self._formation_template, target_center, target_yaw,
            perturbation_std=0.0,  # No perturbation on targets
        )
        return target_positions

    ################################################################################
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with formation-based initial and target positions.
        
        Each episode:
        1. Random formation center within arena bounds
        2. Random formation yaw orientation
        3. Small perturbation on each drone position
        4. Random target center (same formation shape) at a reachable distance
        """
        # Re-generate formation-based initial positions
        self.INIT_XYZS = self._generate_formation_positions()
        
        # Re-generate formation-based target positions
        self.TARGET_POS = self._generate_target_positions()
        
        # Call parent reset which will use the updated INIT_XYZS
        return super().reset(seed=seed, options=options)
    
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # Check if ALL drones reached their targets
        # Using 0.05m (5cm) threshold per drone (matching DMPC-Swarm)
        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
            if dist > 0.05:  # 5cm threshold per drone
                return False
        return True  # All drones within 5cm of targets

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 5.0 or abs(states[i][1]) > 5.0 or states[i][2] > 5.0 # Truncate when a drone is too far away
             or abs(states[i][7]) > 1.0 or abs(states[i][8]) > 1.0 # Truncate when a drone is too tilted (57 deg)
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
