from gym_drill.envs.Coordinate import Coordinate

class TargetBall():
    def __init__(self, x, y, rad):
        self.center = Coordinate(x,y)
        self.radius = rad
        
        self.reached = False # Indicate that the target has been reached.