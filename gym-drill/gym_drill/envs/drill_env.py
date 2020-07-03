import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

# DEFINE OUR CONSTANTS
SCREEN_X = 1000
SCREEN_Y = 600

DRILL_SPEED = 3.0
# TODO: Adjust these values relative to DRILL_SPEED
MAX_INCLINATION = 3.0 * (np.pi / 180)
MAX_ANG_VEL = 0.3 * (np.pi / 180)
MAX_ANG_ACC = 0.05 * (np.pi / 180)

# The allowed increment. We either add or remove this value to the angular acceleration
ANG_ACC_INCREMENT = 0.01 * (np.pi / 180)


HIT_REWARD = 1000 # Reward for hitting a target ball
APPROACH_REW = DRILL_SPEED * 7 # Rew. for approaching tball.
MOVE_REW = DRILL_SPEED * -1    # Penalty for each move.
STEER_REW = -5    # Penalty for altering ang. velocity.
OOF_REW = -10000 # Out of field reward (leaving the screen)


TARGET_MIN_X = SCREEN_X * 0.05
TARGET_MAX_X = SCREEN_X * 0.95

TARGET_MIN_Y = SCREEN_Y * 0.35
TARGET_MAX_Y = SCREEN_Y * 0.95

TARGET_MIN_RAD = 20
TARGET_MAX_RAD = 50


# This is hard-coded in! Don't change:
N_TARGETS_IN_OBS_SPACE = 2

# Targetball class
class TargetBall():
    def __init__(self, x, y, rad):
        self.pos = np.array([x, y])
        self.rad = rad


# Helper functions

def generate_random_drill_x_value():
    # This will generate a number from a gamma-distribution
    # with the values k=5, phi=1.0.
    x = -1
    while not TARGET_MIN_X < x < TARGET_MAX_X:
        # Magic number alert
        x = np.random.gamma(5, 1.0) * (SCREEN_X / 20)
    return x

def generate_targets():
    # Generate a random list of targets.
    # The order of the targets in the list determines the order
    # in which the drill will hit them.

    target_list = []

    # Let's just create three targets.
    # TODO: This function has to be improved.
    x1 = np.random.uniform(0.5*TARGET_MAX_X, TARGET_MAX_X*0.9)
    y1 = np.random.uniform(0.1*TARGET_MAX_Y, 0.5*TARGET_MAX_Y)
    rad1 = np.random.uniform(TARGET_MIN_RAD, TARGET_MAX_RAD)

    x2 = np.random.uniform(0.5*TARGET_MAX_X, TARGET_MAX_X*0.9)
    y2 = np.random.uniform(0.1*TARGET_MAX_Y, 0.5*TARGET_MAX_Y)
    rad2 = np.random.uniform(TARGET_MIN_RAD, TARGET_MAX_RAD)

    x3 = np.random.uniform(0.5*TARGET_MAX_X, TARGET_MAX_X*0.9)
    y3 = np.random.uniform(0.1*TARGET_MAX_Y, 0.5*TARGET_MAX_Y)
    rad3 = np.random.uniform(TARGET_MIN_RAD, TARGET_MAX_RAD)

    
    target_list.append(TargetBall(x1, y1, rad1))
    target_list.append(TargetBall(x2, y2, rad2))
    target_list.append(TargetBall(x3, y3, rad3))

    return target_list

def angle_between_vectors(v1, v2):
    # Return angle in radians between two 2D vectors
    rad_ang = np.math.atan2(np.linalg.det([v1,v2]), np.dot(v1,v2))
    return rad_ang

def dist_between(pos1, pos2):
    # Return distance between pos1 and pos2
    # Both must be numpy vectors
    return np.linalg.norm(pos1 - pos2)


class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        # Define action and observation space.
        # drill_x, drill_y, inclination, ang_vel, ang_acc, targ1_x, targ1_y, targ1_rad, targ2_x, targ2_y, targ2_rad
        self.obs_space_low  = np.array([
            0, 0, 0, -MAX_ANG_VEL, -MAX_ANG_ACC,
            0, 0, TARGET_MIN_RAD, 0, 0, TARGET_MIN_RAD
            ])
        self.obs_space_high = np.array([
            SCREEN_X, SCREEN_Y, 2*np.pi, MAX_ANG_VEL, MAX_ANG_ACC,
            SCREEN_X, SCREEN_Y, TARGET_MAX_RAD, SCREEN_X, SCREEN_Y, TARGET_MAX_RAD
            ])

        # TODO: What data type is optimal?
        # TODO: What data type is big enough to store values?
        self.observation_space = spaces.Box(self.obs_space_low, self.obs_space_high, dtype=np.float64)
        self.action_space = spaces.Discrete(3)

        # Create state array, and fill it with values
        self.state = np.zeros_like(self.obs_space_low)
        self.reset()


        self.viewer = None
        self.seed
        self.target_render_list = []
        self.step_render_list = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_state(self):
        self.state = np.array([
            self.pos[0], self.pos[1], self.inclination, self.ang_vel, self.ang_acc,
            self.targets[0].pos[0], self.targets[0].pos[1], self.targets[0].rad,
            self.targets[1].pos[0], self.targets[1].pos[1], self.targets[1].rad
        ])

    def step(self, action):

        # Update all attribites related to position
        self.update_pos(action)

        # Check if we have hit any targets;
        # calculate reward.
        reward, done = self.calculate_reward()

        return self.state, reward, done, {}


    def calculate_reward(self):
        reward = 0
        done = False

        # If drill left the screen, then give negative reward, and end episode.
        if not (0 < self.pos[0] < SCREEN_X and 0 < self.pos[1] < SCREEN_Y):
            reward += OOF_REW
            done = True
            return reward, done

        # Give a negative reward for moving
        reward += MOVE_REW

        # Give a negative reward if angular acceleration is not zero
        if self.ang_acc != 0:
            reward += STEER_REW

        # We have a target. It is the first targetball in the list.

        # If the target has been hit:
        if dist_between(self.pos, self.targets[0].pos) < self.targets[0].rad:

            # Remove target from target list
            self.targets.pop()
        
            # Give a big one time reward.
            reward += HIT_REWARD

            # If we now have hit all the targets, we return
            if len(self.targets) == 0:
                done = True
                return reward, done


            # If not, we will now shift the targets in the observation space

            # If we now only have one target remaining, duplicate it in obs. space:
            if len(self.targets) == N_TARGETS_IN_OBS_SPACE - 1:
                # Overwrite the target we just hit 
                self.state[5] = self.state[8]
                self.state[6] = self.state[9]
                self.state[7] = self.state[10]
            
            # Else we need to shift the targets in the obs. space
            else:
                # Get the position of the next target
                next_target = self.targets[0]

                self.state[5] = self.state[8]
                self.state[6] = self.state[9]
                self.state[7] = self.state[10]

                self.state[8]  = next_target.pos[0]
                self.state[9]  = next_target.pos[1]
                self.state[10] = next_target.rad

        # If we have not yet hit the current target,
        # give a reward for approaching it.
        else:
            # Find the vector that points from drill and to target
            # We also have the heading vector
            # The angle between these two vectors decides the reward

            # The reward is multiplied by 0 if angle is pi
            # 1 if 0*pi
            # -1 if -1*pi degree
            approach_vector = self.targets[0].pos - self.pos
            inclination_vector = np.array([np.sin(self.inclination), np.cos(self.inclination)])
            angle = angle_between_vectors(approach_vector, inclination_vector)

            reward_factor = np.cos(angle)
            reward += reward_factor * APPROACH_REW
        
        self.update_state()
        return reward, done

    def update_pos(self, action):
        # Update angular acceleration, if within limits
        if action == 0 and self.ang_acc > -MAX_ANG_ACC:
            self.ang_acc -= ANG_ACC_INCREMENT
        elif action == 1 and self.ang_acc < MAX_ANG_ACC:
            self.ang_acc += ANG_ACC_INCREMENT

        # Update angular velocity, if within limits
        if abs(self.ang_vel + self.ang_acc) < MAX_ANG_VEL:
            self.ang_vel += self.ang_acc

        # Update heading, if within limits
        if abs(self.inclination + self.ang_vel) < MAX_INCLINATION:
            self.inclination += self.ang_vel
            self.inclination %= (2*np.pi)

        # Update position
        self.pos[0] += DRILL_SPEED * np.sin(self.inclination)
        self.pos[1] += DRILL_SPEED * np.cos(self.inclination)

        self.update_state()

        # Add position to position-tracking list
        # TODO: This probably work, but not 100% certain.
        self.step_render_list.append(self.pos)
        
    def reset(self, x = -1, y = -1, inclination = -1, targets = []):
        self.viewer = None
        # Reset the state of the environment to an initial state.
        # If x = -1, reset it to random value

        # Set start angular velocity and angular acceleration.
        self.ang_vel = 0
        self.ang_acc = 0

        # Set start position. If no value give, choose one at random.
        # self.pos is a 2D array with x- and y-coordinates
        self.pos = np.array([0, 0])
        if x == -1 or y == -1:
            self.pos[0] = generate_random_drill_x_value()
            self.pos[1] = np.random.uniform(0.95*SCREEN_Y, 1.0*SCREEN_Y)
        else:
            self.pos[0] = x
            self.pos[1] = y

        # Set inclination (heading). If no value given, choose one at random.
        if inclination == -1:
            self.inclination = 3.5*np.pi/4 #np.random.normal(0, np.pi/6) % (2*np.pi)
        else:
            self.inclination = inclination
        
        # Initialize targets if none are given.
        if len(targets) == 0:
            self.targets = generate_targets()
        else:
            self.targets = targets
        
        # Clear path tracking list
        self.step_render_list = []
        
        # Add the positions and radii of all targets to the render list, so we can draw them later.
        # This is a list containing numpy 3D vectors: [x-pos, y-pos, radius] for each target.
        self.target_render_list = [] # Clear list, just in case.
        for target in self.targets:
            self.target_render_list.append(np.array([target.pos[0], target.pos[1], target.rad]))

        # drill_x, drill_y, inclination, ang_vel, ang_acc, targ1_x, targ1_y, targ1_rad, targ2_x, targ2_y, targ2_rad
        self.state = np.array([
            self.pos[0], self.pos[1], self.inclination, self.ang_vel, self.ang_acc,
            self.targets[0].pos[0], self.targets[0].pos[1], self.targets[0].rad,
            self.targets[1].pos[0], self.targets[1].pos[1], self.targets[1].rad
        ])
        self.close()
        return self.state

        

    def render(self, mode='human'):
        screen_width = SCREEN_X
        screen_height = SCREEN_Y
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.bittrans = rendering.Transform()

            self.dbit = rendering.make_circle(6)
            self.dbit.set_color(0.5, 0.5, 0.5)
            self.dbit.add_attr(self.bittrans)
            self.viewer.add_geom(self.dbit)

            for target in self.target_render_list:
                self.tballtrans = rendering.Transform()

                self.tball = rendering.make_circle(target[2])
                self.tball.set_color(0, 0, 0)
                self.tball.add_attr(self.tballtrans)
                self.viewer.add_geom(self.tball)
                self.tballtrans.set_translation(target[0], target[1])
        
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


    def display_environment(self):
        # Get path from step_render_list
        drill_x_pos_list = []
        drill_y_pos_list = []
        for pos in self.step_render_list:
            drill_x_pos_list.append(pos[0])
            drill_y_pos_list.append(pos[1])

        # Plot circles from targetballs
        theta = np.linspace(0, 2*np.pi, 100)
        for target in self.target_render_list:
            rad = target[2]

            x = target[0] + rad*np.cos(theta)
            y = target[1] + rad*np.sin(theta)
            plt.plot(x, y, "r")
        
        # Set axes
        axes = plt.gca()
        axes.set_xlim(0, SCREEN_X)
        axes.set_ylim(0, SCREEN_Y)

        plt.plot(drill_x_pos_list, drill_y_pos_list, "b")
        plt.title("Well trajectory path")

        plt.show()

