"""
A place to implement smaller custom support classes to be used in the environment
"""
import numpy as np

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

def isWithinTraget(bitPosition,targetPosition,targetRadius):
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 < targetRadius

def all_visited(list):
    for i in range(len(list)):
        if list[i]==False:
            return False
    return True
