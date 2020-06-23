import gym
from gym import spaces
from gym.utils import seeding
from customAdditions import Coordinate

import numpy as np

# ---------- Global constant vars for class ---------- #
# Max values for angular velocity and acceleration
MAX_HEADING = 3.0
MAX_ANGVEL = 0.5
MAX_ANGACC = 0.1

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01

# Screen size
SCREEN_X = 600.0
SCREEN_Y = 400.0

# ---------- For testing, should be removed ---------- #
BIT_SPEED = 5.0

START_LOCATION = Coordinate(100.0, SCREEN_Y - 20.0)

TARTGET_LOCATION = Coordinate(500,100)
TARGET_RADIUS = 30

BIT_INITIALIZATION = [0.0,0.0,0.0]

class DrillEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,startLocation,targetLocation,targetRadius,bitInitialization):
        # X and Y position of drill bit
        self.bitLocation = startLocation

        # Current heading in angles of drill bit
        self.heading = bitInitialization[0]
        self.angVel = bitInitialization[1]
        self.angAcc = bitInitialization[2]

        self.targetLocation = targetLocation
        self.targetRadius = targetRadius

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, -MAX_ANGVEL]), np.array([50, 50, 359.9, MAX_ANGVEL]), dtype=np.float32)

        self.seed()
    
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
        self.x += BIT_SPEED * np.sin(self.heading)
        self.y += BIT_SPEED * np.cos(self.heading)

        # If drill is no longer on screen, game over.
        if not (0 < self.x < SCREEN_X and 0 < self.y < SCREEN_Y):
            reward = -1000.0
            done = True
        
        # Check if targetball hit
        if np.linalg.norm([self.ball_x - self.x, self.ball_y - self.y]) < self.ball_rad:
            reward = 100.0
            done = True

        self.state = (self.x, self.y, self.heading, self.angVel, self.angAcc)

        return np.array(self.state), reward, done, {}

    def reset(self,startLocation,bitInitialization):
        self.bitLocation = startLocation

        self.heading = bitInitialization[0]
        self.angVel = bitInitialization[1]
        self.angAcc = bitInitialization[2]

        self.state = (self.x, self.y, self.heading, self.angVel, self.angAcc)
        return np.array(self.state)


    def render(self, mode='human'):
        screen_width = SCREEN_X
        screen_height = SCREEN_Y

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Create drill bit
            self.bittrans = rendering.Transform()

            self.dbit = rendering.make_circle(20)
            self.dbit.set_color(0, 0, 0)
            self.dbit.add_attr(self.bittrans)
            self.viewer.add_geom(self.dbit)

            self.tracing_list = []

            # Draw target ball
            self.tballtrans = rendering.Transform()

            self.tball = rendering.make_circle(self.ball_rad)
            self.tball.set_color(0, 0, 0)
            self.tball.add_attr(self.tballtrans)
            self.viewer.add_geom(self.tball)
            self.tballtrans.set_translation(self.ball_x, self.ball_y)


        this_state = self.state
        self.bittrans.set_translation(this_state[0], this_state[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None