"""
A place to implement smaller custom support functions to be used in the environment
"""
import numpy as np

from gym_drill.envs.Coordinate import Coordinate
#from gym_drill.envs.Target import TargetBall


def isWithinTraget(bitPosition,targetPosition,targetRadius):
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 < targetRadius

def all_visited(list):
    for i in range(len(list)):
        if list[i]==False:
            return False
    return True





