import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
#import customAdditions as ca

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
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 < targetRadius**2
        
# Max values for angular velocity and acceleration
MAX_HEADING = 3.0
MAX_ANGVEL = 0.05
MAX_ANGACC = 0.1

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01

# Screen size
SCREEN_X = 600
SCREEN_Y = 400

DRILL_SPEED = 5.0

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):      
        self.viewer = None      

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, -MAX_ANGVEL, -MAX_ANGACC]), np.array([50, 50, 359.9, MAX_ANGVEL, MAX_ANGACC]), dtype=np.float32)

        self.seed()

    def initParameters(self,startLocation,targets,bitInitialization):

        self.start_x = startLocation.x
        self.start_y = startLocation.y
        
        # We init parameters here        
        self.bitLocation = startLocation
        self.heading = bitInitialization[0]
        self.angVel = bitInitialization[1]
        self.angAcc = bitInitialization[2]

        # For resetting the environment
        self.initialBitLocation = startLocation
        self.initialHeading = bitInitialization[0]
        self.initialAngVel = bitInitialization[1]
        self.initialAngAcc = bitInitialization[2]

        # List containing lists of point and radius of targets
        self.targets = targets
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        reward = -1.0
        done = False

        # Update angular acceleration, if within limits
        if action == 0 and self.angAcc > -MAX_ANGACC:
            self.angAcc -= ANGACC_INCREMENT
        elif action == 1 and self.angAcc < MAX_ANGACC:
            self.angAcc += ANGACC_INCREMENT

        # Update angular velocity, if within limits
        if abs(self.angVel + self.angAcc) < MAX_ANGVEL:
            self.angVel += self.angAcc

        # Update heading, if within limits
        if abs(self.heading + self.angVel) < MAX_HEADING:
            self.heading += self.angVel

        # Update position
        self.bitLocation.x += DRILL_SPEED * np.sin(self.heading)
        self.bitLocation.y += DRILL_SPEED * np.cos(self.heading)

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < SCREEN_X and 0 < self.bitLocation.y < SCREEN_Y):
            reward = -1000.0
            done = True
        
        # Check if targetball hit


        for target in self.targets:
            if isWithinTraget(self.bitLocation,target[0],target[1]):
                reward = 100.0
                done = True

        self.state = (self.bitLocation.x, self.bitLocation.y, self.heading, self.angVel, self.angAcc)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y

        self.heading = self.initialHeading
        self.angVel = self.initialAngVel
        self.angAcc = self.initialAngAcc

        self.state = (self.start_x, self.start_y, self.heading, self.angVel, self.angAcc)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = SCREEN_X
        screen_height = SCREEN_Y
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            #from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Create drill bit
            self.bittrans = rendering.Transform()

            self.dbit = rendering.make_circle(6)
            self.dbit.set_color(0.5, 0.5, 0.5)
            self.dbit.add_attr(self.bittrans)
            self.viewer.add_geom(self.dbit)


            # Draw target ball

            for target in self.targets:
                targetCenter = target[0]
                targetRadius = target[1]

                self.tballtrans = rendering.Transform()

                self.tball = rendering.make_circle(targetRadius)
                self.tball.set_color(0, 0, 0)
                self.tball.add_attr(self.tballtrans)
                self.viewer.add_geom(self.tball)
                self.tballtrans.set_translation(targetCenter.x, targetCenter.y)


        # Update position of drill on screen
        this_state = self.state
        self.bittrans.set_translation(this_state[0], this_state[1])

        # Every iteration add a new tracing point
        self.new_trans = rendering.Transform()
        self.new_trans.set_translation(this_state[0], this_state[1])

        self.new_point = rendering.make_circle(2)
        self.new_point.set_color(0.5, 0.5, 0.5)
        self.new_point.add_attr(self.new_trans)

        self.viewer.add_geom(self.new_point)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None