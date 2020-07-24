from gym.envs.registration import register
from gym_drill.envs.Coordinate import Coordinate
import numpy as np

register(
    id='drill-v0',
    entry_point='gym_drill.envs:DrillEnv',
    kwargs={
        # Default values
        "startLocation" : Coordinate(1000,1000,0),
        "bitInitialization" : [np.random.uniform(0,2*np.pi),np.random.uniform(0,np.pi/4), 0.0, 0.0, 0.0, 0.0]

    }
)