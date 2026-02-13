import numpy as np

from onpolicy.envs.gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from onpolicy.envs.gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

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
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a multi-agent RL environment.

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

        """
        self.EPISODE_LEN_SEC = 8
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
        
        # RANDOMIZED Target positions for generalization
        # Each target is 0.3-1.0 meters away (must be reachable within episode)
        self.TARGET_POS = np.zeros_like(self.INIT_XYZS)
        for i in range(num_drones):
            distance = np.random.uniform(0.3, 1.0)    # Reachable distance
            angle = np.random.uniform(0, 2 * np.pi)   # Random horizontal direction
            z_offset = np.random.uniform(0.2, 0.8)    # Moderate vertical offset
            
            self.TARGET_POS[i] = self.INIT_XYZS[i] + [
                distance * np.cos(angle),  # X offset
                distance * np.sin(angle),  # Y offset
                z_offset                   # Z offset
            ]

    ################################################################################
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with RANDOMIZED initial and target positions.
        
        This ensures the model learns general navigation skills, not memorization.
        """
        # Randomize initial positions before reset
        for i in range(self.NUM_DRONES):
            self.INIT_XYZS[i] = [
                np.random.uniform(-1.5, 1.5),  # Random X
                np.random.uniform(-1.5, 1.5),  # Random Y
                np.random.uniform(0.1, 0.5)     # Random Z (above ground)
            ]
        
        # Randomize target positions relative to new initial positions
        # Keep distances reachable: 0.3-1.0m (max speed ~0.83 m/s, 8s episode)
        for i in range(self.NUM_DRONES):
            distance = np.random.uniform(0.3, 1.0)    # Reachable distance
            angle = np.random.uniform(0, 2 * np.pi)   # Random horizontal direction
            z_offset = np.random.uniform(0.2, 0.8)    # Moderate vertical offset
            
            self.TARGET_POS[i] = self.INIT_XYZS[i] + [
                distance * np.cos(angle),  # X offset
                distance * np.sin(angle),  # Y offset
                z_offset                   # Z offset
            ]
        
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
