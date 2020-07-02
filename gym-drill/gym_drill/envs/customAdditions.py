"""
A place to implement smaller custom support classes to be used in the environment
"""
from gym_drill.envs.Coordinate import Coordinate

def isWithinTraget(bitPosition,targetPosition,targetRadius):
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 < targetRadius

def all_visited(list):
    for i in range(len(list)):
        if list[i]==False:
            return False
    return True
