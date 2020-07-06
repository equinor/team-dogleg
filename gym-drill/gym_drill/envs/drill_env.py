import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

# Our own libs
from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.ObservationSpace import ObservationSpace
from gym_drill.envs.Target import TargetBall
import gym_drill.envs.environment_support as es

# Max values for angular velocity and acceleration
MAX_HEADING = 3.0 # issue1: In the obs space we set this to 360 deg
MAX_ANGVEL = 0.05
MAX_ANGACC = 0.1

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01
DRILL_SPEED = 5.0

# Screen size, environment should be square
SCREEN_X = 600
SCREEN_Y = 600

# Observation space specs
SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y] # x_low,x_high,y_low,y_high
BIT_BOUNDS = [0,2*np.pi,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGACC,MAX_ANGACC] #

# Target specs
TARGET_BOUND_X = [0.5*SCREEN_X,0.9*SCREEN_X]
TARGET_BOUND_Y = [0.1*SCREEN_Y,0.6*SCREEN_Y]
TARGET_RADII_BOUND = [20,50]

NUM_TARGETS = 4
TARGET_WINDOW_SIZE = 3

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,startLocation,bitInitialization):
        self.start_x = startLocation.x
        self.start_y = startLocation.y
        # Save the starting position as "first" step. Needed for plotting in matplotlib
        self.step_history = [[self.start_x,self.start_y]]        

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

        # Init targets. See _init_targets function
        self.targets = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,startLocation)
        
        self.action_space = spaces.Discrete(3)        
         
        self.observation_space_container= ObservationSpace(SPACE_BOUNDS,BIT_BOUNDS,self.targets)
      
        self.observation_space = self.observation_space_container.get_space_box3()        

        self.seed()
        self.viewer = None      
  
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):

        done = False        

        self.update_bit(action)

        reward = -1.0 #step-penalty

        # Maybe create an entire function that handles all rewards, and call it here?
        if self.angAcc != 0:
            reward -= 1.0 #angAcc-penalty

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < SCREEN_X and 0 < self.bitLocation.y < SCREEN_Y):
            reward  -=1000.0
            done = True     
                
        self.state = self.get_state()

        return np.array(self.state), reward, done, {}
    
    # For encapsulation. Updates the bit according to the action
    def update_bit(self,action):
        # Update angular acceleration, if within limits
        if action == 0 and self.angAcc > -MAX_ANGACC:
            self.angAcc -= ANGACC_INCREMENT
        elif action == 1 and self.angAcc < MAX_ANGACC:
            self.angAcc += ANGACC_INCREMENT

        # Update angular velocity, if within limits
        if abs(self.angVel + self.angAcc) < MAX_ANGVEL:
            self.angVel += self.angAcc

        # Update heading, if within limits
        if abs(self.heading + self.angVel) < MAX_HEADING: # issue1
            self.heading += self.angVel

        # Update position
        self.bitLocation.x += DRILL_SPEED * np.sin(self.heading)
        self.bitLocation.y += DRILL_SPEED * np.cos(self.heading)
        self.step_history.append([self.bitLocation.x,self.bitLocation.y])

    # Returns tuple of current state
    def get_state(self):
        state_list = [self.bitLocation.x, self.bitLocation.y, self.heading, self.angVel, self.angAcc]
        for target in self.observation_space_container.target_window: # This will cause bug
            state_list.append(target.center.x)
            state_list.append(target.center.y)
            state_list.append(target.radius)

        return tuple(state_list)        

    def reset(self):
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y

        self.heading = self.initialHeading
        self.angVel = self.initialAngVel
        self.angAcc = self.initialAngAcc

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y]]       

        # Need to init new targets
        self.targets = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,self.bitLocation)             

        self.state = self.get_state()

        return np.array(self.state)

    
    """
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
    """

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def display_environment(self):
        # Get data
        x_positions = []
        y_positions = []
        for position in self.step_history:
            x_positions.append(position[0])
            y_positions.append(position[1])       

        
        # Plot circles from targetballs, just to verify the order of the balls
        theta = np.linspace(0, 2*np.pi, 100)
        colors_order = {
            1:"b",
            2:"g",
            3:"r",
            4:"c",
            5:"m",
            6:"y",
            7:"k"
            }
        cnt = 1
        t = 0
        for target in self.targets:
            t += 1
            center = target.center
            radius = target.radius
            
            d = Coordinate.getEuclideanDistance(Coordinate(x_positions[0],y_positions[0]),target.center)
            print("Distance from start to target #",t, "is: ",d)         


            x = center.x + radius*np.cos(theta)
            y = center.y + radius*np.sin(theta)

            plt.plot(x,y,colors_order[cnt])
            cnt += 1                
        
        # Set axis 
        axes = plt.gca()
        axes.set_xlim(0,SCREEN_X)
        axes.set_ylim(0,SCREEN_Y)

        plt.plot(x_positions,y_positions,"b")
        plt.title("Well trajectory path")

        plt.show()

