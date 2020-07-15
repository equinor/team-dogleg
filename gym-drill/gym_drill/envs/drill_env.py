import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
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
SCREEN_Z = 2000

# Target specs
TARGET_BOUND_X = [0.25*SCREEN_X,0.85*SCREEN_X]
TARGET_BOUND_Y = [0.2*SCREEN_Y,0.75*SCREEN_Y]
TARGET_BOUND_Z = [0.25*SCREEN_Z,0.85*SCREEN_Z]
TARGET_RADII_BOUND = [20,50]

NUM_TARGETS = 4
TARGET_WINDOW_SIZE = 3
NUM_MAX_STEPS = ((SCREEN_X+SCREEN_Y+SCREEN_Z)/DRILL_SPEED)*1.3

# Rewards
FINISHED_EARLY_FACTOR = 1 # Point per unused step

# Hazard specs. Can be in entire screen
HAZARD_BOUND_X = [0,SCREEN_X]
HAZARD_BOUND_Y = [0,SCREEN_Y]
HAZARD_BOUND_Z = [0,SCREEN_Z]
HAZARD_RADII_BOUND = [20,50]

NUM_HAZARDS = 4

# Observation space specs
SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y,0,SCREEN_Z] # x_low,x_high,y_low,y_high
BIT_BOUNDS = [0,2*np.pi,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGACC,MAX_ANGACC] #
HAZARD_BOUNDS = [HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_BOUND_Z,HAZARD_RADII_BOUND]
TARGET_BOUNDS = [TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_BOUND_Z, TARGET_RADII_BOUND]

# Additional data
DIAGONAL = np.sqrt(SCREEN_X**2 + SCREEN_Y**2 + SCREEN_Z**2)
TARGET_DISTANCE_BOUND = [0,DIAGONAL]
RELATIVE_HORIZONTAL_ANGLE_BOUND = [-np.pi,np.pi]
RELATIVE_VERTICAL_ANGLE_BOUND = [-np.pi,np.pi]

EXTRA_DATA_BOUNDS = [TARGET_DISTANCE_BOUND]#,RELATIVE_HORIZONTAL_ANGLE_BOUND,RELATIVE_VERTICAL_ANGLE_BOUND ] # [Distance, angle between current direction and target direction]

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,startLocation,bitInitialization,*,activate_hazards=False):
        self.start_x = startLocation.x
        self.start_y = startLocation.y
        self.start_z = startLocation.z
        # Save the starting position as "first" step. Needed for plotting in matplotlib
        self.step_history = [[self.start_x,self.start_y,self.start_z]]        

        # We init parameters here        
        self.bitLocation = startLocation
        self.horizontal_heading = uniform(0,np.pi/2)
        self.vertical_heading = uniform(np.pi/10,np.pi/2)

            #self.angVel = bitInitialization[1]
        self.horizontal_angVel = bitInitialization[2]
        self.vertical_angVel =bitInitialization[3]

            #self.angAcc = bitInitialization[2]
        self.horizontal_angAcc = bitInitialization[4]
        self.vertical_angAcc =bitInitialization[5]

       

        # For resetting the environment
        """
        self.initialBitLocation = startLocation
        self.initialHeading = bitInitialization[0]
        self.initialAngVel = bitInitialization[1]
        self.initialAngAcc = bitInitialization[2]
        """

        self.initialBitLocation = startLocation
        
        self.initial_horizontal_heading = bitInitialization[0]
        self.initial_vertical_heading = bitInitialization[1]
        
        self.initial_horizontal_angVel = bitInitialization[2]
        self.initial_vertical_angVel = bitInitialization[3]

        self.initial_horizontal_angAcc = bitInitialization[4]
        self.initial_vertical_angAcc = bitInitialization[5]
        



        # Init targets. See _init_targets function
        self.targets = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_BOUND_Z,TARGET_RADII_BOUND,startLocation)
        self.activate_hazards = activate_hazards
        if self.activate_hazards:
            #print("Initiating environment with hazards")
            self.hazards = es._init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,TARGET_BOUND_Z,HAZARD_RADII_BOUND,startLocation,self.targets)
        else:
            #print("Initiating environment without hazards")
            self.hazards = []

        self.action_space = spaces.Discrete(9)        

        self.observation_space_container= ObservationSpace(SPACE_BOUNDS,TARGET_BOUNDS,HAZARD_BOUNDS,BIT_BOUNDS,EXTRA_DATA_BOUNDS,self.targets,self.hazards)
      
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
        reward, done = self.get_reward_and_done_signal()           

        self.state = self.get_state()
        #self.total_reward += reward
        return np.array(self.state), reward, done, {}

    
    # Returns the reward for the step and if episode is over
    def get_reward_and_done_signal(self):
        done = False      
        reward = 0.0 #step-penalty
        
        # Maybe create an entire function that handles all rewards, and call it here?
        """
        if self.angAcc != 0:
            reward -= 2.0 #angAcc-penalty
        
        if self.angVel != 0:
            reward -= 1.0 #angAcc-penalty
        """

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < SCREEN_X and 0 < self.bitLocation.y < SCREEN_Y and 0 < self.bitLocation.z < SCREEN_Z):
            reward  -=30
            done = True   
        
        # Check if we hit a hazard
        for h in self.hazards:
            if es._is_within(self.bitLocation,h.center,h.radius):
                reward -= 100.0
                #done = True
                #print("Hazard hit, I will stop")        

        if len(self.step_history)>NUM_MAX_STEPS:
            done= True                        

        # Find the values of the current target
        current_target_pos = np.array([self.state[9], self.state[10], self.state[11]])
        current_target_rad = self.state[12]
        drill_pos = np.array([self.bitLocation.x, self.bitLocation.y, self.bitLocation.z])

        # Check if target is hit
        if np.linalg.norm(current_target_pos - drill_pos) < current_target_rad:
            # If target is hit, give reward.
            reward += 100
            # If we don't have any more targets,
            if len(self.observation_space_container.remaining_targets) == 0:
                # we are done.
                reward += (NUM_MAX_STEPS-len(self.step_history))*FINISHED_EARLY_FACTOR
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

            # Heading vector.  [JUST GIVING IT A TRY DOING THIS FOR EACH ANGLE IN THE 3D CASE]
            
                #vertical-angle
            """
            head_vec = np.array([np.sin(self.horizontal_heading), np.cos(self.horizontal_heading)])
            angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))
            reward_factor = np.cos(angle_between_vectors) # value between -1 and +1 
            reward += reward_factor*4
                #horizontal angle
            head_vec = np.array([np.sin(self.vertical_heading), np.cos(self.vertical_heading)])
            angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))
            reward_factor = np.cos(angle_between_vectors) # value between -1 and +1 
            reward += reward_factor*4
            """
        

        return reward, done
    """
    def get_horizontal_angle_relative_to_target(self):
        current_target = self.observation_space_container.target_window[0]
                
        curr_target_pos_vector = np.array([current_target.center.x,current_target.center.y])

        curr_drill_pos_vector = np.array([self.bitLocation.x,self.bitLocation.y])
        appr_vec = curr_target_pos_vector - curr_drill_pos_vector

        head_vec = np.array([np.sin(self.horizontal_heading), np.cos(self.horizontal_heading)])
        angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))

        return angle_between_vectors

    def get_vertical_angle_relative_to_target(self):
        current_target = self.observation_space_container.target_window[0]
                
        curr_target_pos_vector = np.array([current_target.center.x,current_target.center.z])

        curr_drill_pos_vector = np.array([self.bitLocation.x,self.bitLocation.z])
        appr_vec = curr_target_pos_vector - curr_drill_pos_vector

        head_vec = np.array([np.sin(self.vertical_heading), np.cos(self.vertical_heading)])
        angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))

        return angle_between_vectors
    """
    # For encapsulation. Updates the bit according to the action
    def update_bit(self,action):
        

        # Update angular acceleration, if within limits
        if action < 3 and self.vertical_angAcc < MAX_ANGACC:            #indexes of action space:
            self.vertical_angAcc += ANGACC_INCREMENT                    #   0       1       2 | (0-2): accelerate upwards
        elif action > 5 and self.vertical_angAcc > -MAX_ANGACC:         #   3       4       5 | (3-5): don't accelate in the vertical plane
            self.vertical_angAcc -= ANGACC_INCREMENT                    #   6       7       8 | (6-8): accelerate downwards
                                                                        #---------------------
                                                                        #(0,3,6): accelerate left 
                                                                        #        (1,4,7): don't accelerate in the horizontal plane
                                                                        #                (2,5,8): accelerate right
        if (action == 0) or (action == 3) or (action == 6) and self.horizontal_angAcc > -MAX_ANGACC:
            self.horizontal_angAcc -= ANGACC_INCREMENT
        elif (action == 2) or (action == 5) or (action == 8) and self.horizontal_angAcc < MAX_ANGACC:
            self.horizontal_angAcc += ANGACC_INCREMENT
        

        # Update angular velocity, if within limits

        if abs(self.horizontal_angVel + self.horizontal_angAcc) < MAX_ANGVEL:
            self.horizontal_angVel += self.horizontal_angAcc

        if abs(self.vertical_angVel + self.vertical_angAcc) < MAX_ANGVEL:
            self.vertical_angVel += self.vertical_angAcc


        vertical_speed = abs(np.sin(self.vertical_heading)) * DRILL_SPEED
        horizontal_speed = abs(np.cos(self.vertical_heading)) * DRILL_SPEED

        # Update heading.

        self.vertical_heading = (self.vertical_heading + self.vertical_angVel) % (2 * np.pi)
        self.horizontal_heading = (self.horizontal_heading + self.horizontal_angVel) % (2 * np.pi)

        # Update position
        self.bitLocation.x += horizontal_speed * np.cos(self.horizontal_heading)
        self.bitLocation.y += horizontal_speed * np.sin(self.horizontal_heading)
        self.bitLocation.z += vertical_speed * np.cos(self.vertical_heading)

        self.step_history.append([self.bitLocation.x,self.bitLocation.y, self.bitLocation.z])

    # Returns tuple of current state
    def get_state(self):
        # Core bit data
        state_list = [self.bitLocation.x, self.bitLocation.y, self.bitLocation.z, self.horizontal_heading,self.vertical_heading, self.horizontal_angVel,self.vertical_angVel, self.horizontal_angAcc,self.vertical_angAcc]
        # Target data that are inside the window
        for target in self.observation_space_container.target_window: # This will cause bug
            state_list.append(target.center.x)
            state_list.append(target.center.y)
            state_list.append(target.center.z)
            state_list.append(target.radius)
        # Get all hazards
        for hazard in self.observation_space_container.hazards:
            state_list.append(hazard.center.x)
            state_list.append(hazard.center.y)
            state_list.append(hazard.center.z)
            state_list.append(hazard.radius)
        # Extra data
        current_target = self.observation_space_container.target_window[0]
        distance_to_target = Coordinate.getEuclideanDistance(current_target.center,self.bitLocation)-current_target.radius
        #relative_horizontal_angle = self.get_horizontal_angle_relative_to_target()
        #relative_vertical_angle = self.get_vertical_angle_relative_to_target()

        state_list =  state_list + [distance_to_target]#,relative_horizontal_angle, relative_vertical_angle]
        return tuple(state_list)        

    def reset(self):
        # Save previous run to log
        #self.write_to_log()
        #self.episode_counter += 1
        self.total_reward = 0
        
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y
        self.bitLocation.z = self.start_z

        self.horizontal_heading = uniform(0,np.pi/2)
        self.vertical_heading = uniform(0,np.pi/4)

        self.horizontal_angVel = self.initial_horizontal_angVel
        self.vertical_angVel = self.initial_vertical_angVel

        self.horizontal_angAcc = self.initial_horizontal_angAcc
        self.vertical_angAcc = self.initial_vertical_angAcc
        

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y,self.start_z]]       

        # Need to init new targets
        self.targets = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_BOUND_Z,TARGET_RADII_BOUND,self.bitLocation)             
        
        # Init new hazards
        if self.activate_hazards:
            #print("Initiating environment with hazards")
            self.hazards = es._init_hazards(NUM_HAZARDS,[0.25*SCREEN_X,0.85*SCREEN_X],[0.2*SCREEN_Y,0.75*SCREEN_Y],[0.25*SCREEN_Z,0.85*SCREEN_Z],HAZARD_RADII_BOUND,self.bitLocation,self.targets)
        else:
            #print("Initiating environment without hazards")
            self.hazards = []

        # Re-configure the observation space
        self.observation_space_container= ObservationSpace(SPACE_BOUNDS,TARGET_BOUNDS,HAZARD_BOUNDS,BIT_BOUNDS,EXTRA_DATA_BOUNDS,self.targets,self.hazards)
      
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
    """
    def display_horizontal_plane_of_environment(self):
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
    
    def display_vertical_plane_of_environment(self):
        # Get data
        x_positions = []
        z_positions = []
        for position in self.step_history:
            x_positions.append(position[0])
            z_positions.append(position[2])      

        
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
            z = center.z + radius*np.sin(theta)
            label = "Target #" + str(cnt)
            plt.plot(x,z,colors_order[cnt],label=label)
            cnt += 1

        firsttime = True # To ensure hazard label only appears once
        for hazard in self.hazards:
            h_center = hazard.center
            h_radius = hazard.radius
            h_x = h_center.x + h_radius*np.cos(theta)                
            h_z = h_center.z + h_radius*np.sin(theta)
            if firsttime:
                plt.plot(h_x,h_z,"k",label="Hazards")
                firsttime = False
            else:
                plt.plot(h_x,h_z,"k")

        # Set axis 
        axes = plt.gca()
        axes.set_xlim(0,SCREEN_X)
        axes.set_ylim(0,SCREEN_Z)

        plt.plot(x_positions,z_positions,"grey")
        plt.title("Well trajectory path")
        plt.legend()
        plt.show()
    """

    def display_3d_environment(self):
        # Get data
        x_positions = []
        y_positions = []
        z_positions = []
        for position in self.step_history:
            x_positions.append(position[0])
            y_positions.append(position[1])
            z_positions.append(position[2])      

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_positions,y_positions,z_positions)
        #ax.invert_zaxis()

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

            plot_ball(target.center.x,target.center.y,target.center.z,target.radius,colors_order[cnt],ax)
        
                     
            #x = center.x + radius*np.cos(theta)
            #z = center.z + radius*np.sin(theta)
            label = "Target #" + str(cnt)
            #plt.plot(x,y,z,colors_order[cnt],label=label)#?
            
            cnt += 1
        """
        firsttime = True # To ensure hazard label only appears once
        for hazard in self.hazards:
            h_center = hazard.center
            h_radius = hazard.radius
            h_x = h_center.x + h_radius*np.cos(theta)                
            h_z = h_center.z + h_radius*np.sin(theta)
            if firsttime:
                plt.plot(h_x,h_z,"k",label="Hazards")
                firsttime = False
            else:
                plt.plot(h_x,h_z,"k")
                """
        for hazard in self.hazards:
            plot_ball(hazard.center.x,hazard.center.y,hazard.center.z,hazard.radius,'k',ax)


        # Set axis 
        #ax = plt.gca()
        ax.set_xlim(SCREEN_X,0)
        ax.set_ylim(0,SCREEN_Y)
        ax.set_zlim(SCREEN_Z,0)
        
        ax.set_xlabel("North")
        ax.set_ylabel("East")
        ax.set_zlabel("Down")

        return fig
        """
        plt.plot(x_positions,z_positions,"grey")
        plt.title("Well trajectory path")
        plt.legend()
        plt.show()
        """
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

def plot_ball(x0,y0,z0,r,c,ax):
    
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))
    # Plot the surface
    ax.plot_surface(x, y, z, color=c)

if __name__ == '__main__':
    print("Testing init of targets and hazards")    
    startpos = Coordinate(100,400,100)

    print("Creating targets")
    t = es._init_targets(NUM_TARGETS,TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_BOUND_Z,TARGET_RADII_BOUND,startpos)
    for _ in t:
        print(_)
    
    print("Creating Hazards")    
    h = es._init_hazards(NUM_HAZARDS,HAZARD_BOUND_X,HAZARD_BOUND_Y,TARGET_BOUND_Z,HAZARD_RADII_BOUND,startpos,t)
    for eden_hazard in h:
        print(eden_hazard) #haha
    
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
    
    print("Verify Environemnt")
    import random
    BIT_INITIALIZATION = [3.5*np.pi/4,np.pi/2, 0.0, 0.0, 0.0, 0.0]

    env = DrillEnv(startpos,BIT_INITIALIZATION)

    action_size = env.action_space.n
    action = random.choice(range(action_size))
    env.step(action)
    print("I took one step, this is what the current state is:")
    print(env.state)
    print(len(env.state))
    print(env.observation_space)

   