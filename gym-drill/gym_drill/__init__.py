from gym.envs.registration import register
from gym_drill.envs.Coordinate import Coordinate
import numpy as np

"""
# A cartesian coordinate
class Coordinate:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y
    
    # For displaying the coordinate
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    # For boolean comparison (if myPoint == yourPoint)
    def __eq__(self, other):
        return self.y == other.y and self.x == other.x

    @classmethod
    def getEuclideanDistance(cls,p1,p2):
        return np.abs(np.sqrt((p1.x - p2.x)**2 + (p1.y-p2.y)**2))
"""

register(
    id='drill-v0',
    entry_point='gym_drill.envs:DrillEnv',
    kwargs={
        # Default values
        "startLocation" : Coordinate(1000,1000,0),
        "bitInitialization" : [3*np.pi/4,0.0,0.0]

    }
)