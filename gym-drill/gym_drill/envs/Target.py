from gym_drill.envs.Coordinate import Coordinate

class TargetBall():
    def __init__(self, x, y, z, rad):
        self.center = Coordinate(x,y,z)
        self.radius = rad
        
        self.reached = False # Indicate that the target has been reached.
        self.reward = 100

    # Print the target to console. Can be nice for debugging purposes
    def __str__(self):
        return "Center:" + str(self.center) +". Radii:" + str(self.radius)





