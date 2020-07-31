from gym_drill.envs.Coordinate import Coordinate

HAZARD_PENALTY = -200 #?
# Currently Hazard is the opposite of reward
class Hazard:
    def __init__(self,x,y,z,rad):
        self.center = Coordinate(x,y,z)
        self.radius = rad

<<<<<<< HEAD:gym-drill/gym_drill/envs/Hazard.py
        self.penalty = HAZARD_PENALTY
        self.is_hit = False
=======
        self.is_hit = False

>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/Hazard.py
    def __str__(self):
        return "Center:" + str(self.center) +". Radii:" + str(self.radius)