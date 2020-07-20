from gym_drill.envs.Coordinate import Coordinate

HAZARD_PENALTY = -100 #?
# Currently Hazard is the opposite of reward
class Hazard:
    def __init__(self,x,y,z,rad):
        self.center = Coordinate(x,y,z)
        self.radius = rad

        self.penalty = HAZARD_PENALTY
        self.is_hit = False
    def __str__(self):
        return "Center:" + str(self.center) +". Radii:" + str(self.radius)