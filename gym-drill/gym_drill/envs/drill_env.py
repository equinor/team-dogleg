import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
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

def all_visited(list):
    for i in range(len(list)):
        if list[i]==False:
            return False
    return True

# Max values for angular velocity and acceleration
MAX_HEADING = 3.0
MAX_ANGVEL = 0.05
MAX_ANGACC = 0.1

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01

# Screen size, environment should be square
SCREEN_X = 600
SCREEN_Y = 600

# Target specs
TARGET_BOUND_X =[0.5*SCREEN_X,0.9*SCREEN_X]
TARGET_BOUND_Y = [0.1*SCREEN_Y,0.6*SCREEN_Y]
TARGET_RADII_BOUND = [20,50]

DRILL_SPEED = 5.0
NUM_TARGETS = 2

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,startLocation,bitInitialization):
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

        # Init targets. List containing lists of targets of random radius and position
        self.targets = []
        self.visited = []
        for target in range(NUM_TARGETS):
            target_center = Coordinate(np.random.uniform(TARGET_BOUND_X[0],TARGET_BOUND_X[1]),(np.random.uniform(TARGET_BOUND_Y[0],TARGET_BOUND_Y[1] )))
            target_radius = np.random.uniform(TARGET_RADII_BOUND[0],TARGET_RADII_BOUND[1])

            target_pair = [target_center,target_radius]
            self.targets.append(target_pair)
            self.visited.append(False)

        self.viewer = None      

        self.action_space = spaces.Discrete(3)

        # Init observation space
        lower_obs_space_limit = np.array([0, 0, 0, -MAX_ANGVEL, -MAX_ANGACC])
        upper_obs_space_limit = np.array([SCREEN_X,SCREEN_Y, 2*np.pi, MAX_ANGVEL, MAX_ANGACC])

        for target in range(NUM_TARGETS):
            lower_obs_space_limit = np.append(lower_obs_space_limit,[TARGET_BOUND_X[0],TARGET_BOUND_Y[0],TARGET_RADII_BOUND[0]])
            upper_obs_space_limit = np.append(upper_obs_space_limit,[TARGET_BOUND_X[1],TARGET_BOUND_Y[1],TARGET_RADII_BOUND[1]])

        #Can be made as a seperate function
        self.initial_distances=[]
        for i in range (NUM_TARGETS):
            self.initial_distances.append(Coordinate.getEuclideanDistance(self.bitLocation,self.targets[i][0]))
        #print(self.initial_distances)
        
        
        self.observation_space = spaces.Box(lower_obs_space_limit,upper_obs_space_limit, dtype=np.float64)
        print("The length of the observation space is:",len(lower_obs_space_limit))
        
        self.seed()

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y]]       
  
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
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
        self.step_history.append([self.bitLocation.x,self.bitLocation.y])

        reward = -1.0 #step-penalty

        if self.angAcc != 0:
            reward -= 1.0 #angAcc-penalty

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < SCREEN_X and 0 < self.bitLocation.y < SCREEN_Y):
            reward  -=1000.0
            done = True
        
        # Check if targetball hit  
        i=0
        for target in self.targets:
            if isWithinTraget(self.bitLocation,target[0],target[1]) and self.visited[i]==False: 
                self.visited[i]=True
                reward +=1000
                if all_visited(self.visited):
                    done = True
                    #reward += 150 - number of steps taken
            i+=1

                
        state_list = [self.bitLocation.x, self.bitLocation.y, self.heading, self.angVel, self.angAcc]
        for target in self.targets:
            state_list.append(target[0].x)
            state_list.append(target[0].y)
            state_list.append(target[1])

        self.state = tuple(state_list)

        return np.array(self.state), reward, done, {}


    def reset(self):
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y

        self.heading = self.initialHeading
        self.angVel = self.initialAngVel
        self.angAcc = self.initialAngAcc

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y]]       

        # List containing lists of targets of random radius and position
        self.targets = []
        for target in range(NUM_TARGETS):
            target_center = Coordinate(np.random.uniform(TARGET_BOUND_X[0],TARGET_BOUND_X[1]),(np.random.uniform(TARGET_BOUND_Y[0],TARGET_BOUND_Y[1] )))
            target_radius = np.random.uniform(TARGET_RADII_BOUND[0],TARGET_RADII_BOUND[1])

            target_pair = [target_center,target_radius]
            self.targets.append(target_pair)            

        state_list = [self.bitLocation.x, self.bitLocation.y, self.heading, self.angVel, self.angAcc]
        for target in self.targets:
            state_list.append(target[0].x)
            state_list.append(target[0].y)
            state_list.append(target[1])

        self.state = tuple(state_list)        
    
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

    def display_environment(self):
        # Get data
        x_positions = []
        y_positions = []
        for position in self.step_history:
            x_positions.append(position[0])
            y_positions.append(position[1])
        
        # Plot circles from targetballs
        theta = np.linspace(0, 2*np.pi, 100)
        for target in self.targets:
            center = target[0]
            radius = target[1]

            x = center.x + radius*np.cos(theta)
            y = center.y + radius*np.sin(theta)

            plt.plot(x,y,"r")            
        
        # Set axis 
        axes = plt.gca()
        axes.set_xlim(0,SCREEN_X)
        axes.set_ylim(0,SCREEN_Y)

        plt.plot(x_positions,y_positions,"b")
        plt.title("Well trajectory path")

        plt.show()
