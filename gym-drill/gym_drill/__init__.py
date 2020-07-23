from gym.envs.registration import register
import numpy as np
from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs import environment_config as cfg

register(
    id='drill-v0',
    entry_point='gym_drill.envs:DrillEnv',
    kwargs={
        # Default values
        "startLocation" : Coordinate(cfg.SCREEN_X*0.1,cfg.SCREEN_Y*0.8),
        "bitInitialization" : [np.random.uniform(np.pi/2,np.pi),0.0,0.0]

    }
)