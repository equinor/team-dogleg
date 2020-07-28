from gym.envs.registration import register
from gym_drill.envs.Coordinate import Coordinate
import gym_drill.envs.environment_config as cfg
import numpy as np

register(
    id='drill-v0',
    entry_point='gym_drill.envs:DrillEnv',
    # Default values
    kwargs={
        # Starting at the top, center of the screen
        "startLocation" : Coordinate(0.5*cfg.SCREEN_X,0.5*cfg.SCREEN_Y,0),
        # Random Azimuth and Inclination angle. Angular velocity and acceleration of both are set to zero
        "bitInitialization" : [np.random.uniform(0,2*np.pi),np.random.uniform(0,np.pi/4), 0.0, 0.0, 0.0, 0.0]
    }
)