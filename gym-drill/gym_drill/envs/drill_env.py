import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

# Our own libs
from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.ObservationSpace import ObservationSpace
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.Hazard import Hazard

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

# Target specs
TARGET_BOUND_X = [0.5*SCREEN_X,0.9*SCREEN_X]
TARGET_BOUND_Y = [0.1*SCREEN_Y,0.6*SCREEN_Y]
TARGET_RADII_BOUND = [20,50]

NUM_TARGETS = 4
TARGET_WINDOW_SIZE = 3

# Hazard specs. Can be in entire screen
HAZARD_BOUND_X = [0,SCREEN_X]
HAZARD_BOUND_Y = [0,SCREEN_Y]
HAZARD_RADII_BOUND = [20,50]

NUM_HAZARDS = 4

# Observation space specs
SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y] # x_low,x_high,y_low,y_high
BIT_BOUNDS = [0,2*np.pi,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGACC,MAX_ANGACC] #
HAZARD_BOUNDS = [HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND]
TARGET_BOUNDS = [TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND]

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,startLocation,bitInitialization,*,activate_hazards=False):
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
        
        if activate_hazards:
            print("Initiating environment with hazards")
            self.hazards = _init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND,startLocation,self.targets)
        else:
            print("Initiating environment without hazards")
            self.hazards = []

        self.action_space = spaces.Discrete(3)        

        self.observation_space_container= ObservationSpace(SPACE_BOUNDS,TARGET_BOUNDS,HAZARD_BOUNDS,BIT_BOUNDS,self.targets,self.hazards)
      
        self.observation_space = self.observation_space_container.get_space_box()        

        self.seed()
        self.viewer = None
        self.state = self.get_state()      
  
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
                

        # Find the values of the current target
        current_target_pos = np.array([self.state[5], self.state[6]])
        current_target_rad = self.state[7]
        drill_pos = np.array([self.bitLocation.x, self.bitLocation.y])

        # Check if target is hit
        if np.linalg.norm(current_target_pos - drill_pos) < current_target_rad:
            # If target is hit, give reward.
            reward += 1000
            # If we don't have any more targets,
            if len(self.observation_space_container.remaining_targets) == 0:
                # we are done.
                done = True

            # But if we do have more targets,
            else:
                # we must shift the targets.
                self.observation_space_container.shift_window()

        
        else:
            # If target is not hit, then we give a reward if drill is approaching it.
            # Find the vector that points from drill and to target
            # We also have the heading vector
            # The angle between these two vectors decides the reward

            # The reward is multiplied by 0 if angle is pi
            # 1 if 0*pi
            # -1 if -1*pi degree

            # Approach vector
            appr_vec = current_target_pos - drill_pos
            # Heading vector.
            head_vec = np.array([np.sin(self.heading), np.cos(self.heading)])
            angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))
            reward_factor = np.cos(angle_between_vectors)
            reward += reward_factor * 7       

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
        for hazard in self.observation_space_container.hazards:
            state_list.append(hazard.center.x)
            state_list.append(hazard.center.y)
            state_list.append(hazard.radius)

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

        
        # Plot circles from targetballs, colors just to verify the order of the balls
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
        for target in self.targets:
            center = target.center
            radius = target.radius          
                           
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


# Returns an ordered list of randomly generated targets within the bounds given. 
def _init_targets(num_targets,x_bound,y_bound,r_bound,start_location):
    all_targets = []

    for t in range(num_targets):
        target = _create_unique_random_target(start_location,x_bound,y_bound,r_bound,all_targets)
        all_targets.append(target)        
    
    all_targets = _orderTargets(start_location,all_targets)

    return all_targets

def _init_hazards(num_hazards,x_bound,y_bound,r_bound,start_pos,existing_targets):
    all_hazards = []
    for h in range(num_hazards):
        avoid = existing_targets + all_hazards
        hazard = _create_unique_random_hazard(start_pos,x_bound,y_bound,r_bound,avoid)
        all_hazards.append(hazard)

    return all_hazards

# Finds nearest between 1 point and a list of candidate points
# startlocation is type Coordinate, and candidates is list of types Targets
def _findNearest(start_location,candidates):
    current_shortest_distance = -1 # Init with an impossible distance
    current_closest_target_index = 0
    for candidate_index in range(len(candidates)):        
        candidate = candidates[candidate_index]     
        distance = Coordinate.getEuclideanDistance(candidate.center,start_location)
        
        if distance < current_shortest_distance or current_shortest_distance == -1:           
            current_shortest_distance = distance
            current_closest_target_index = candidate_index
        
    return current_closest_target_index

# Orders the target based upon a given start location
# start_location is type Coordinate, all_targets is list of type targets
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

# Returns True if t1 or t2 overlap. Works for both Hazards and Targets
def _is_overlapping(t1,t2):
    total_radii = t1.radius + t2.radius
    distance = Coordinate.getEuclideanDistance(t1.center,t2.center)
    return  distance < total_radii

# Creates a uniqe target that does not overlap with any targets in existing_targets
def _create_unique_random_target(start_pos,x_bound,y_bound,r_bound,existing_targets):
    target_center = Coordinate(np.random.uniform(x_bound[0],x_bound[1]),(np.random.uniform(y_bound[0],y_bound[1] )))
    target_radius = np.random.uniform(r_bound[0],r_bound[1])
    target_candidate = TargetBall(target_center.x,target_center.y,target_radius)

    for target in existing_targets:
        if _is_overlapping(target,target_candidate) or _isWithin(start_pos,target_center,target_radius):
            target_candidate =_create_unique_random_target(start_pos,x_bound,y_bound,r_bound,existing_targets)
            break

    return target_candidate

# Creates a uniqe hazard that does not overlad with any obstacles in existing_obstacles
def _create_unique_random_hazard(start_pos,x_bound,y_bound,r_bound,existing_obstacles):
    hazard_center = Coordinate(np.random.uniform(x_bound[0],x_bound[1]),(np.random.uniform(y_bound[0],y_bound[1] )))
    hazard_radius = np.random.uniform(r_bound[0],r_bound[1])
    hazard_candidate = Hazard(hazard_center.x,hazard_center.y,hazard_radius)  
    
    for obstacle in existing_obstacles:
        if _is_overlapping(obstacle,hazard_candidate) or _isWithin(start_pos,hazard_center,hazard_radius):
            hazard_candidate = _create_unique_random_hazard(start_pos,x_bound,y_bound,r_bound,existing_obstacles)
            break
    
    return hazard_candidate

def _isWithin(bitPosition,targetPosition,targetRadius):
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 < targetRadius

if __name__ == '__main__':
    print("Testing init of targets and hazards")    
    startpos = Coordinate(100,400)

    print("Creating targets")
    t = _init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND,startpos)
    for _ in t:
        print(_)
    
    print("Creating Hazards")    
    h = _init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND,startpos,t)
    for eden_hazard in h:
        print(eden_hazard)
    
    # Plot circles from targetballs, colors just to verify the order of the balls
    theta = np.linspace(0, 2*np.pi, 100)
    colors_order = {
        1:"b",
        2:"g",
        3:"r",
        4:"c",
        5:"m",
        6:"y",        
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
    
    print("Verify Environemnt")
    import random
    BIT_INITIALIZATION = [3.5*np.pi/4,0.0,0.0]

    env = DrillEnv(startpos,BIT_INITIALIZATION)

    action_size = env.action_space.n
    action = random.choice(range(action_size))
    env.step(action)
    print("I took one step, this is what the current state is:")
    print(env.state)
    print(len(env.state))
    print(env.observation_space)

   