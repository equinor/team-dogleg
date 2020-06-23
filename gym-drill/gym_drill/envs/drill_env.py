import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

STATESPACE_POS = 50
STATESPACE_HEADING = 8
STATESPACE_ANGVEL = 3

# Max values for angular velocity and acceleration
MAX_HEADING = 3.0
MAX_ANGVEL = 0.5
MAX_ANGACC = 0.1

ANGACC_INCREMENT = 0.01

# Screen size
SCREEN_X = 600
SCREEN_Y = 400

DRILL_SPEED = 5.0
DRILL_X0 = 100.0
DRILL_Y0 = SCREEN_Y - 20.0

BALL_X = 500
BALL_Y = 100
BALL_RAD = 30


class DrillEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        # X and Y position of drill bit
        self.x = DRILL_X0
        self.y = DRILL_Y0

        # Current heading in angles of drill bit
        self.heading = 0.0
        self.angVel = 0.0 #  First derivative of heading
        self.angAcc = 0.0 # Second derivative of heading

        self.ball_x = BALL_X
        self.ball_y = BALL_Y
        self.ball_rad = BALL_RAD


        self.viewer = None

        self.low = np.array([0, 0, 0, -MAX_ANGVEL])
        self.high = np.array([50, 50, 359.9, MAX_ANGVEL])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

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
        self.x += DRILL_SPEED * np.sin(self.heading)
        self.y += DRILL_SPEED * np.cos(self.heading)

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

    def reset(self):
        self.x = DRILL_X0
        self.y = DRILL_Y0

        self.heading = 0.0
        self.angVel = 0.0
        self.angAcc = 0.0

        self.state = (self.x, self.y, self.heading, self.angVel, self.angAcc)
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
            self.tballtrans = rendering.Transform()

            self.tball = rendering.make_circle(self.ball_rad)
            self.tball.set_color(0, 0, 0)
            self.tball.add_attr(self.tballtrans)
            self.viewer.add_geom(self.tball)
            self.tballtrans.set_translation(self.ball_x, self.ball_y)


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