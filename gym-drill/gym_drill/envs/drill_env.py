import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

# Our own libs
from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.ObservationSpace import ObservationSpace 
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.customAdditions import *

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
        self.targets = _init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,startLocation)
        
        self.action_space = spaces.Discrete(3)        
         
        self.observation_space_container = ObservationSpace(SPACE_BOUNDS,BIT_BOUNDS,self.targets)
        print("here is the obs spacer")
        print(self.observation_space_container)        
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
        for target in self.targets:
            state_list.append(target.center.x)
            state_list.append(target.center.y)
            state_list.append(target.radius)

        self.state = tuple(state_list) 

    def reset(self):
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y

        self.heading = self.initialHeading
        self.angVel = self.initialAngVel
        self.angAcc = self.initialAngAcc

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y]]       

        # Need to init new targets
        self.targets = _init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,self.bitLocation)             

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
        

        """
        # Plot circles from targetballs
        theta = np.linspace(0, 2*np.pi, 100)
        for target in self.targets:
            center = target[0]
            radius = target[1]

            x = center.x + radius*np.cos(theta)
            y = center.y + radius*np.sin(theta)

            plt.plot(x,y,"r")
        """            
        
        # Set axis 
        axes = plt.gca()
        axes.set_xlim(0,SCREEN_X)
        axes.set_ylim(0,SCREEN_Y)

        plt.plot(x_positions,y_positions,"b")
        plt.title("Well trajectory path")

        plt.show()

# Finds nearest between 1 point and a list of candidate points
# startlocation is type Coordinate, and candidates is list of types Targets
def _findNearest(start_location,candidates):
    current_shortest_distance = SCREEN_X # Distance cant be greater than the screen
    current_closest_target_index = 0
    for candidate_index in range(len(candidates)):        
        candidate = candidates[candidate_index]     
        distance = Coordinate.getEuclideanDistance(candidate.center,start_location)
        if distance < current_shortest_distance:
            current_shortest_distance = distance
            current_closest_target_index = candidate_index
    
    return candidate_index

# Orders the target based upon a given start location
# start_location is type Coordiante, all_targets is list of type targets
def _orderTargets(start_location,all_targets):
    #target_order = [None] * len(all_targets) # Maybe better with = [] and use append()
    target_order = [] 
    loop_counter = 0
    while len(all_targets) != 0:
        if loop_counter == 0:
            next_in_line_index = _findNearest(start_location,all_targets)
            next_in_line_target = all_targets[next_in_line_index]
            target_order.append(next_in_line_target)
            all_targets.pop(next_in_line_index)
        else:
            next_in_line_index = _findNearest(target_order[loop_counter-1].center,all_targets)
            next_in_line_target = all_targets[next_in_line_index]
            target_order.append(next_in_line_target)
            all_targets.pop(next_in_line_index)
        
        loop_counter += 1

    return target_order

# Returns an ordered list of randomly generated targets within the bounds given. 
def _init_targets(num_targets,x_bound,y_bound,r_bound,start_location):
    all_targets = []

    for t in range(num_targets):
        target_center = Coordinate(np.random.uniform(x_bound[0],x_bound[1]),(np.random.uniform(y_bound[0],y_bound[1] )))


        target_radius = np.random.uniform(r_bound[0],r_bound[1])

        target = TargetBall(target_center.x,target_center.y,target_radius)
               
        all_targets.append(target)
    
    all_targets = _orderTargets(start_location,all_targets)

    return all_targets

# Will 
def _init_observation_space(window_targets):
    # These are all given by global vars in the environment
    lower_obs_space_limit = np.array([0, 0, 0, -MAX_ANGVEL, -MAX_ANGACC])
    upper_obs_space_limit = np.array([SCREEN_X,SCREEN_Y, 2*np.pi, MAX_ANGVEL, MAX_ANGACC])

    for target in range(len(window_targets)):
        pass
