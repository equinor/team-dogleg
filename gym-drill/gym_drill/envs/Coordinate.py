"""
Simple Coordinate classs
"""
import numpy as np

# A cartesian coordinate
class Coordinate:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z
    
    # For displaying the coordinate
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," +str(self.z) + ")"

    # For boolean comparison (if myPoint == yourPoint)
    def __eq__(self, other):
        return self.y == other.y and self.x == other.x and self.z == other.z

    @classmethod
    def getEuclideanDistance(cls,p1,p2):
        return np.abs(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2))