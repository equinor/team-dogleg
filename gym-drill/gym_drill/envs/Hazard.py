from gym_drill.envs.Coordinate import Coordinate

HAZARD_PENALTY = -100 #?
# Currently Hazard is the opposite of reward
class Hazard:
    def __init__(self,x,y,rad):
        self.center = Coordinate(x,y)
        self.radius = rad

        self.is_hit = False

    def __str__(self):
        return "Center:" + str(self.center) +". Radii:" + str(self.radius)