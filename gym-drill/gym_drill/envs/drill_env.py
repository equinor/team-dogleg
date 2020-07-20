import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from random import uniform

# Our own libs
from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.ObservationSpace import ObservationSpace
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.Hazard import Hazard
from gym_drill.envs import environment_support as es

# Max values for angular velocity and acceleration
MAX_ANGVEL = 0.1
MAX_ANGACC = 0.05

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01
DRILL_SPEED = 5.0

# Screen size, environment should be square
SCREEN_X = 2000
SCREEN_Y = 2000

# Target specs
TARGET_BOUND_X = [0.25*SCREEN_X,0.85*SCREEN_X]
TARGET_BOUND_Y = [0.2*SCREEN_Y,0.75*SCREEN_Y]
TARGET_RADII_BOUND = [40,70]

NUM_TARGETS = 6

# Hazard specs. Can be in entire screen
HAZARD_BOUND_X = [0,SCREEN_X]
HAZARD_BOUND_Y = [0,SCREEN_Y]
HAZARD_RADII_BOUND = [40,70]

NUM_HAZARDS = 6 # MUST BE EQUAL OR GREATER THAN HAZARD WINDOW SIZE

# Observation space specs
SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y] # x_low,x_high,y_low,y_high
BIT_BOUNDS = [0,2*np.pi,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGACC,MAX_ANGACC] #
HAZARD_BOUNDS = [HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND]
TARGET_BOUNDS = [TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND]

# Additional data
DIAGONAL = np.sqrt(SCREEN_X**2 + SCREEN_Y**2)
TARGET_DISTANCE_BOUND = [0,DIAGONAL]
RELATIVE_ANGLE_BOUND = [-np.pi,np.pi]
EXTRA_DATA_BOUNDS = [TARGET_DISTANCE_BOUND,RELATIVE_ANGLE_BOUND] # [Distance, angle between current direction and target direction]

# All reward values go here. The reward will add these values. Make sure signs are correct!
STEP_PENALTY = -1.0
ANGULAR_VELOCITY_PENALTY = 0.0
ANGULAR_ACCELERATION_PENALTY = 0.0
OUTSIDE_SCREEN_PENALTY = -50.0
TARGET_REWARD = 100.0
HAZARD_PENALTY = -200.0
ANGLE_REWARD_FACTOR = 2

NUM_MAX_STEPS = ((SCREEN_X+SCREEN_Y)/DRILL_SPEED)*1.3
FINISHED_EARLY_FACTOR = 2 # Point per unused step

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,startLocation,bitInitialization,*,activate_hazards=True):
        self.start_x = startLocation.x
        self.start_y = startLocation.y
        # Save the starting position as "first" step. Needed for plotting in matplotlib
        self.step_history = [[self.start_x,self.start_y]]        

        # We init parameters here        
        self.bitLocation = startLocation
        self.heading = uniform(np.pi/2,np.pi)
        self.angVel = bitInitialization[1]
        self.angAcc = bitInitialization[2]

        # For resetting the environment
        self.initialBitLocation = startLocation
        self.initialHeading = bitInitialization[0]
        self.initialAngVel = bitInitialization[1]
        self.initialAngAcc = bitInitialization[2]

        # Init targets. See _init_targets function
        self.targets = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,startLocation)
        self.activate_hazards = activate_hazards
        if self.activate_hazards:
            self.hazards = es._init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND,startLocation,self.targets)
        else:
            self.hazards = []

        self.action_space = spaces.Discrete(3)        

        self.observation_space_container= ObservationSpace(SPACE_BOUNDS,TARGET_BOUNDS,HAZARD_BOUNDS,BIT_BOUNDS,EXTRA_DATA_BOUNDS,self.targets,self.hazards,self.bitLocation)
      
        self.observation_space = self.observation_space_container.get_space_box()        

        self.seed()
        self.viewer = None
        self.state = self.get_state()

        # Log related
        """
        self.episode_counter = 0 # Used to write to log
        self.total_reward = 0      
        es._init_log()
        """

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        self.update_bit(action)
        self.observation_space_container.update_hazard_window(self.bitLocation)
        reward, done = self.get_reward_and_done_signal()           

        self.state = self.get_state()
        #self.total_reward += reward
        return np.array(self.state), reward, done, {}

    
    # Returns the reward for the step and if episode is over
    def get_reward_and_done_signal(self):
        done = False      
        reward = STEP_PENALTY
        
        # Maybe create an entire function that handles all rewards, and call it here?
        if self.angAcc != 0:
            reward += ANGULAR_ACCELERATION_PENALTY

        if self.angVel != 0:
            reward += ANGULAR_VELOCITY_PENALTY

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < SCREEN_X and 0 < self.bitLocation.y < SCREEN_Y):
            reward  += OUTSIDE_SCREEN_PENALTY
            done = True   
        
        # Check if we hit a hazard
        for h in self.observation_space_container.hazard_window:
            if es._is_within(self.bitLocation,h.center,h.radius) and not h.is_hit:
                reward += HAZARD_PENALTY 
                h.is_hit = True
                #done = True
                
        if len(self.step_history)>NUM_MAX_STEPS:
            done= True                        

        # Find the values of the current target
        current_target_pos = np.array([self.state[5], self.state[6]])
        current_target_rad = self.state[7]
        drill_pos = np.array([self.bitLocation.x, self.bitLocation.y])

        # Check if target is hit
        if np.linalg.norm(current_target_pos - drill_pos) < current_target_rad:
        #if es._is_within(self.bitLocation,self.observation_space_container.target_window[0].center,self.observation_space_container.target_window[0].radius):
            reward += TARGET_REWARD
            if len(self.observation_space_container.remaining_targets) == 0:
                reward += (NUM_MAX_STEPS-len(self.step_history))*FINISHED_EARLY_FACTOR
                done = True
            else:
                self.observation_space_container.shift_target_window()
        
        else:
            # Approach vector
            appr_vec = current_target_pos - drill_pos

            # Heading vector.
            head_vec = np.array([np.sin(self.heading), np.cos(self.heading)])
            angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))
            reward_factor = np.cos(angle_between_vectors) # value between -1 and +1 
            #adjustment =(1-abs(10*self.angVel))**3
            # adjustment = 0 if angVel = +-MAX      #adjustment = 1 if angVel = 0
            reward += reward_factor*ANGLE_REWARD_FACTOR# * adjustment 
        

        return reward, done
    def get_angle_relative_to_target(self):
        current_target = self.observation_space_container.target_window[0]
                
        curr_target_pos_vector = np.array([current_target.center.x,current_target.center.y])

        curr_drill_pos_vector = np.array([self.bitLocation.x,self.bitLocation.y])
        appr_vec = curr_target_pos_vector - curr_drill_pos_vector

        head_vec = np.array([np.sin(self.heading), np.cos(self.heading)])
        angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))

        return angle_between_vectors
    
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

        # Update heading.
        self.heading = (self.heading + self.angVel) % (2 * np.pi)

        # Update position
        self.bitLocation.x += DRILL_SPEED * np.sin(self.heading)
        self.bitLocation.y += DRILL_SPEED * np.cos(self.heading)
        self.step_history.append([self.bitLocation.x,self.bitLocation.y])

    # Returns tuple of current state
    def get_state(self):
        # Core bit data
        state_list = [self.bitLocation.x, self.bitLocation.y, self.heading, self.angVel, self.angAcc]
        # Target data that are inside the window
        for target in self.observation_space_container.target_window: # This will cause bug
            state_list.append(target.center.x)
            state_list.append(target.center.y)
            state_list.append(target.radius)
        # Get hazards inside window
        for hazard in self.observation_space_container.hazard_window:
            state_list.append(hazard.center.x)
            state_list.append(hazard.center.y)
            state_list.append(hazard.radius)
        # Extra data
        current_target = self.observation_space_container.target_window[0]
        distance_to_target = Coordinate.getEuclideanDistance(current_target.center,self.bitLocation)-current_target.radius
        relative_angle = self.get_angle_relative_to_target() 

        state_list =  state_list + [distance_to_target,relative_angle]
        return tuple(state_list)        

    def reset(self):
        # Save previous run to log
        #self.write_to_log()
        #self.episode_counter += 1
        self.total_reward = 0
        
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y

        self.heading = uniform(np.pi/2,np.pi)
        self.angVel = self.initialAngVel
        self.angAcc = self.initialAngAcc

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y]]       

        # Need to init new targets
        self.targets = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,self.bitLocation)             
        
        # Init new hazards
        if self.activate_hazards:
            self.hazards = es._init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND,self.bitLocation,self.targets)
        else:
            self.hazards = []

        # Re-configure the observation space
        self.observation_space_container= ObservationSpace(SPACE_BOUNDS,TARGET_BOUNDS,HAZARD_BOUNDS,BIT_BOUNDS,EXTRA_DATA_BOUNDS,self.targets,self.hazards,self.bitLocation)
      
        self.observation_space = self.observation_space_container.get_space_box()        
        
        self.state = self.get_state()
        
        return np.array(self.state)
    
    """
    def write_to_log(self,*,filename="drill_log.txt"):
        f = open(filename,"a")
        text = "Episode nr: " +str(self.episode_counter) + " lasted for " + str(len(self.step_history)) + " steps. My total reward was: " + str(self.total_reward)  +"\n"
        
        f.write(text)
        f.close()
        #print("Log updated!")
        """
   
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def display_state(self):
        print("Bit location:", Coordinate(self.state[0],self.state[1]))
        print("Bit angles: ", self.state[2:5])
        print("Targets inside window: ")
        for i in range(len(self.observation_space_container.target_window)):
            t = TargetBall(self.state[5+3*i],self.state[6+3*i],self.state[7+3*i])
            print(t)
        print("Hazards inside window")
        for i in range(len(self.observation_space_container.hazard_window)):
            h = Hazard(self.state[5+3*i],self.state[6+3*i],self.state[7+3*i])
            print(h)

        print("Extra data:")
        print("Distance: ",self.state[5+3*len(self.observation_space_container.target_window)+ 3*len(self.observation_space_container.hazard_window)])
        print("Relative angle: ",self.state[5+3*len(self.observation_space_container.target_window)+ 3*len(self.observation_space_container.hazard_window)+1])

    def display_environment(self):
        # Get data
        x_positions = []
        y_positions = []
        for position in self.step_history:
            x_positions.append(position[0])
            y_positions.append(position[1])      

        
        # Plot circles from targetballs, colors just to verify the order of the balls
        theta = np.linspace(0, 2*np.pi, 100)
        colors_order = {
            1:"b",
            2:"g",
            3:"r",
            4:"c",
            5:"m",
            6:"y",
            7:"palevioletred",
            8:"pink",
            9:"coral",
            10:"orange",
            11:"saddlebrown"
            }
        cnt = 1
        for target in self.targets:
            center = target.center
            radius = target.radius          
                           
            x = center.x + radius*np.cos(theta)
            y = center.y + radius*np.sin(theta)
            label = "Target #" + str(cnt)
            plt.plot(x,y,colors_order[cnt],label=label)
            cnt += 1

        firsttime = True # To ensure hazard label only appears once
        for hazard in self.hazards:
            h_center = hazard.center
            h_radius = hazard.radius
            h_x = h_center.x + h_radius*np.cos(theta)                
            h_y = h_center.y + h_radius*np.sin(theta)
            if firsttime:
                plt.plot(h_x,h_y,"k",label="Hazards")
                firsttime = False
            else:
                plt.plot(h_x,h_y,"k")

        # Set axis 
        axes = plt.gca()
        axes.set_xlim(0,SCREEN_X)
        axes.set_ylim(0,SCREEN_Y)

        plt.plot(x_positions,y_positions,"grey")
        plt.title("Well trajectory path")
        plt.legend()
        plt.show()
      
    
if __name__ == '__main__':
    startpos = Coordinate(100,900)
    """
    print("Testing init of targets and hazards")    

    print("Creating targets")
    t = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,startpos)
    for _ in t:
        print(_)
    
    print("Creating Hazards")    
    h = es._init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND,startpos,t)
    for eden_hazard in h:
        print(eden_hazard)
    
    # plot circles from targetballs, colors just to verify the order of the balls
    theta = np.linspace(0, 2*np.pi, 100)
    colors_order = {
        1:"b",
        2:"g",
        3:"r",
        4:"c",
        5:"m",
        6:"y",
        7:"palevioletred",
        8:"pink",
        9:"coral",
        10:"orange",
        11:"saddlebrown"        
        }
    cnt = 1
    for target in t:
        center = target.center
        radius = target.radius          
                           
        x = center.x + radius*np.cos(theta)
        y = center.y + radius*np.sin(theta)

        plt.plot(x,y,colors_order[cnt])
        cnt += 1
    for hazard in h:
        h_center = hazard.center
        h_radius = hazard.radius
        h_x = h_center.x + h_radius*np.cos(theta)                
        h_y = h_center.y + h_radius*np.sin(theta)
        plt.plot(h_x,h_y,"k")

    # Set axis 
    axes = plt.gca()
    axes.set_xlim(0,SCREEN_X)
    axes.set_ylim(0,SCREEN_Y)
    
    plt.title("Test random generated hazard and targets")
    plt.show()
    """
    print("Verify Environemnt")
    import random
    BIT_INITIALIZATION = [np.pi/2,0.0,0.0]

    env = DrillEnv(startpos,BIT_INITIALIZATION,activate_hazards=True)

    action_size = env.action_space.n
    action = random.choice(range(action_size))
    env.step(action)
    print("I took one step, this is what the current state is:")
    print(env.state)
    env.observation_space_container.display_hazards()
    env.display_environment()
    
    for _ in range (5000):
        action = random.choice(range(action_size))
        env.step(action)
    print("50 steps later")
    env.display_state()
    env.observation_space_container.display_hazards()
    env.display_environment()
   
    print("Resetting")
    env.reset()
    env.display_state()
    env.observation_space_container.display_hazards()
    env.display_environment()