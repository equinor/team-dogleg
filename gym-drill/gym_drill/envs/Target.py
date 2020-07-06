from gym_drill.envs.Coordinate import Coordinate

class TargetBall():
    def __init__(self, x, y, rad):
        self.center = Coordinate(x,y)
        self.radius = rad
        
        self.reached = False # Indicate that the target has been reached.
        self.name = "Have no name yet"
    
    def set_name(self,name):
        self.name = name
        
    # Print the target to console. Can be nice for debugging purposes
    def __str__(self):
        return "Center:" + str(self.center) +". Radii:" + str(self.radius)




