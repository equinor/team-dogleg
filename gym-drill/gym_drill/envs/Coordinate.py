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
        return np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]))


if __name__ == "__main__":
    v1 = Coordinate(3, 3, 10)
    v2 = Coordinate(-4, 2, 4)
    dist = Coordinate.getEuclideanDistance(v1, v2)
    if 0.99 < dist/9.273618495495704 < 1.01:
        print("getEuclideanDistance seems to be working")
    else:
        print("getEuclideanDistance doesnt work")
    
