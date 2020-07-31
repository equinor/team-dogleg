import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
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
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
from gym_drill.envs import rdwellpath as rwp
from gym_drill.envs import environment_config as cfg
=======
from gym_drill.envs import environment_config as cfg

ENVIRONMENT_FILENAME = "environments.txt"
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
}

    def __init__(self,startLocation,bitInitialization,*,activate_hazards=True,monte_carlo=False,activate_log=False,load=False):
        self.activate_log = activate_log
        self.activate_hazards = activate_hazards
        self.monte_carlo = monte_carlo

<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        self.start_x = startLocation.x
        self.start_y = startLocation.y
        self.start_z = startLocation.z
=======
    def __init__(self,startLocation,bitInitialization,*,activate_hazards=True,random_envs=True):
        self.activate_hazards = activate_hazards
        self.random_envs = random_envs
        self.start_x = startLocation.x
        self.start_y = startLocation.y        
        
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
        # Save the starting position as "first" step. Needed for plotting in matplotlib
        self.step_history = [[self.start_x,self.start_y,self.start_z]]        

        # We init parameters here        
        self.bitLocation = startLocation
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        self.azimuth_heading = bitInitialization[0]
        self.inclination_heading = bitInitialization[1]

        self.azimuth_angVel = bitInitialization[2]
        self.inclination_angVel =bitInitialization[3]

        self.azimuth_angAcc = bitInitialization[4]
        self.inclination_angAcc =bitInitialization[5]      

        # For resetting the environment        
        self.initial_azimuth_heading = bitInitialization[0]
        self.initial_inclination_heading = bitInitialization[1]        
        self.initial_azimuth_angVel = bitInitialization[2]
        self.initial_inclination_angVel = bitInitialization[3]
        self.initial_azimuth_angAcc = bitInitialization[4]
        self.initial_inclination_angAcc = bitInitialization[5]        
        
        # Generate feasible environments to train in using a Monte Carlo simulation 
        if self.monte_carlo and not load:
            print("Running", str(cfg.NUM_MONTE_CARLO_ENVS),"Monte Carlo simulations to generate target sets!")
            
            rwp.generate_targets_hazards_to_file(cfg.NUM_TARGETS, cfg.NUM_HAZARDS,
            [cfg.TARGET_BOUND_X[0],cfg.TARGET_BOUND_Y[0],cfg.TARGET_BOUND_Z[0]],
            [cfg.TARGET_BOUND_X[1],cfg.TARGET_BOUND_Y[1],cfg.TARGET_BOUND_Z[1]],
            cfg.MC_PATH_LENGTH_BOUND[0], cfg.MC_PATH_LENGTH_BOUND[1],
            [cfg.TARGET_BOUND_X[0],cfg.TARGET_BOUND_Y[0],cfg.TARGET_BOUND_Z[0]],
            cfg.NUM_MONTE_CARLO_ENVS, cfg.ENVIRONMENT_FILENAME)
        elif load and self.monte_carlo:
            print("Using prexisting Monte Carlo generated environment")
            print("Make sure it matches your trained models setting. See environment.txt for details!")           
        
            
        self.create_targets_and_hazards()
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,self.targets,self.hazards,self.bitLocation)
=======
        self.heading = uniform(np.pi/2,np.pi)
        self.angVel = bitInitialization[1]
        self.angAcc = bitInitialization[2]

        # For resetting the environment
        self.initialHeading = bitInitialization[0]
        self.initialAngVel = bitInitialization[1]
        self.initialAngAcc = bitInitialization[2]

        self.create_targets_and_hazards()

        self.action_space = spaces.Discrete(3)       
        
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,cfg.EXTRA_DATA_BOUNDS,self.targets,self.hazards,self.bitLocation)
      
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
        self.observation_space = self.observation_space_container.get_space_box()        
        self.action_space = spaces.Discrete(9)     

        self.seed()
        self.viewer = None
        self.state = self.get_state()

        # Log related
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        if self.activate_log:
            self.episode_counter = 0 # Used to write to log
            self.total_reward = 0      
            es._init_log()

    def create_targets_and_hazards(self):
        if not self.monte_carlo:
            # Targets are drawn randomly with the target boundaries specified in the config file.
            self.targets = es._init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.TARGET_RADII_BOUND,self.bitLocation)
            if self.activate_hazards:
                self.hazards = es._init_hazards(cfg.NUM_HAZARDS,cfg.HAZARD_BOUND_X,cfg.HAZARD_BOUND_Y,cfg.HAZARD_BOUND_Z,cfg.HAZARD_RADII_BOUND,self.bitLocation,self.targets)
            else:
                self.hazards = []
        else:
            linenr = np.random.randint(1,cfg.NUM_MONTE_CARLO_ENVS-10)
            self.targets,self.hazards = es._read_env_from_file(cfg.ENVIRONMENT_FILENAME,linenr)
            # Overwrite hazards to be empty if not activated
=======
        """
        self.episode_counter = 0 # Used to write to log
        self.total_reward = 0      
        es._init_log()
        """
    def create_targets_and_hazards(self):
        if self.random_envs:
            self.targets = es._init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_RADII_BOUND,self.bitLocation)
            if self.activate_hazards:
                self.hazards = es._init_hazards(cfg.NUM_HAZARDS,cfg.HAZARD_BOUND_X,cfg.HAZARD_BOUND_Y,cfg.HAZARD_RADII_BOUND,self.bitLocation,self.targets)
            else:
                self.hazards = []
        else:
            self.targets,self.hazards = es.read_env_from_file(ENVIRONMENT_FILENAME,2)
            # Overwrite hazards if not activated
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
            if not self.activate_hazards:
                self.hazards = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        self.update_bit(action)
        self.observation_space_container.update_hazard_window(self.bitLocation)
        reward, done = self.get_reward_and_done_signal()           

        self.state = self.get_state()
        return np.array(self.state), reward, done, {}
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
    

=======

    # Returns the reward for the step and if episode is over
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
    def get_reward_and_done_signal(self):
        done = False      
        reward = cfg.STEP_PENALTY
        
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        if self.azimuth_angAcc != 0:
            reward += cfg.ANGULAR_ACCELERATION_PENALTY
        
        if self.azimuth_angVel != 0:
            reward += cfg.ANGULAR_VELOCITY_PENALTY
        
        if self.inclination_angAcc != 0:
            reward += cfg.ANGULAR_ACCELERATION_PENALTY
        
        if self.inclination_angVel != 0:
            reward += cfg.ANGULAR_VELOCITY_PENALTY        

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < cfg.SCREEN_X and 0 < self.bitLocation.y < cfg.SCREEN_Y and 0 < self.bitLocation.z < cfg.SCREEN_Z):
            reward  += cfg.OUTSIDE_SCREEN_PENALTY
            done = True   
        
        # Check if we hit a hazard. atm we do not terminate the episode if we do
        for h in self.observation_space_container.hazard_window:
            if es._is_within(self.bitLocation,h.center,h.radius) and not h.is_hit:
                reward -= cfg.HAZARD_PENALTY
                h.is_hit = True

=======
        # Maybe create an entire function that handles all rewards, and call it here?
        if self.angAcc != 0:
            reward += cfg.ANGULAR_ACCELERATION_PENALTY

        if self.angVel != 0:
            reward += cfg.ANGULAR_VELOCITY_PENALTY

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < cfg.SCREEN_X and 0 < self.bitLocation.y < cfg.SCREEN_Y):
            reward  += cfg.OUTSIDE_SCREEN_PENALTY
            done = True   
        
        # Check if we hit a hazard
        for h in self.observation_space_container.hazard_window:
            if es._is_within(self.bitLocation,h.center,h.radius) and not h.is_hit:
                reward += cfg.HAZARD_PENALTY 
                h.is_hit = True
                #done = True
                
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
        if len(self.step_history)>cfg.NUM_MAX_STEPS:
            done= True                        

        # Find the values of the current target
        current_target = self.observation_space_container.target_window[0]
        current_target_pos = np.array([current_target.center.x, current_target.center.y, current_target.center.z])
        current_target_rad = current_target.radius
        drill_pos = np.array([self.bitLocation.x, self.bitLocation.y, self.bitLocation.z])

        # Check if target is hit
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        #if np.linalg.norm(current_target_pos - drill_pos) < current_target_rad:
        if es._is_within(self.bitLocation,self.observation_space_container.target_window[0].center,self.observation_space_container.target_window[0].radius):
            reward += cfg.TARGET_REWARD

=======
        if np.linalg.norm(current_target_pos - drill_pos) < current_target_rad:
        #if es._is_within(self.bitLocation,self.observation_space_container.target_window[0].center,self.observation_space_container.target_window[0].radius):
            reward += cfg.TARGET_REWARD
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
            if len(self.observation_space_container.remaining_targets) == 0:
                reward += (cfg.NUM_MAX_STEPS-len(self.step_history))*cfg.FINISHED_EARLY_FACTOR
                done = True
            else:
                self.observation_space_container.shift_target_window()
        
        else:
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
            pre_adjusted_azimuth_reward = np.cos(es.get_relative_azimuth_angle(self.bitLocation, self.azimuth_heading, current_target)) # value between -1 and +1 
            reward += pre_adjusted_azimuth_reward*cfg.ANGLE_REWARD_FACTOR
            height_diff = current_target_pos[2]-self.bitLocation.z
            if height_diff != 0:
                reward += np.cos(self.inclination_heading)*(height_diff/abs(height_diff))*cfg.INCLINATION_REWARD_FACTOR #reward for going in the right directen (up/down)
        
        # Log related
        if self.activate_log:
            self.total_reward += reward
=======
            # Approach vector
            appr_vec = current_target_pos - drill_pos

            # Heading vector.
            head_vec = np.array([np.sin(self.heading), np.cos(self.heading)])
            angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))
            reward_factor = np.cos(angle_between_vectors) # value between -1 and +1 
            #adjustment =(1-abs(10*self.angVel))**3
            # adjustment = 0 if angVel = +-MAX      #adjustment = 1 if angVel = 0
            reward += reward_factor*cfg.ANGLE_REWARD_FACTOR# * adjustment 
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py

        return reward, done

    # For encapsulation. Updates the bit according to the action
    def update_bit(self,action):
        #indexes of action space:
        #---------------------#
        #   0       1       2 # (0-2): accelerate upwards
        #   3       4       5 # (3-5): don't accelate in the vertical plane
        #   6       7       8 # (6-8): accelerate downwards
        #---------------------#
        # (0,3,6): accelerate left
        # (1,4,7): don't accelerate in the horizontal plane 
        # (2,5,8): accelerate right
        
        # Update angular acceleration, if within limits
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        # Inclinaition
        if action < 3 and self.inclination_angAcc < cfg.MAX_ANGACC:             
            self.inclination_angAcc += cfg.ANGACC_INCREMENT                     
        elif action > 5 and self.inclination_angAcc > -cfg.MAX_ANGACC:          
            self.inclination_angAcc -= cfg.ANGACC_INCREMENT                
        
        # Azmuth             
        if (action == 0 or action == 3 or action == 6) and self.azimuth_angAcc > -cfg.MAX_ANGACC:
            self.azimuth_angAcc -= cfg.ANGACC_INCREMENT
        elif (action == 2 or action == 5 or action == 8) and self.azimuth_angAcc < cfg.MAX_ANGACC:
            self.azimuth_angAcc += cfg.ANGACC_INCREMENT        

        # Update angular velocity
        # Inclination
        if abs(self.azimuth_angVel + self.azimuth_angAcc) < cfg.MAX_ANGVEL:
            self.azimuth_angVel += self.azimuth_angAcc

        elif (self.azimuth_angVel + self.azimuth_angAcc) <= -cfg.MAX_ANGVEL:
            self.azimuth_angVel = -cfg.MAX_ANGVEL
            self.azimuth_angAcc = 0
        
        elif (self.azimuth_angVel + self.azimuth_angAcc) >= cfg.MAX_ANGVEL:
            self.azimuth_angVel = cfg.MAX_ANGVEL
            self.azimuth_angAcc = 0

        # Azimuth
        if abs(self.inclination_angVel + self.inclination_angAcc) < cfg.MAX_ANGVEL:
            self.inclination_angVel += self.inclination_angAcc

        elif (self.inclination_angVel + self.inclination_angAcc) <= -cfg.MAX_ANGVEL:
            self.inclination_angVel = -cfg.MAX_ANGVEL
            self.inclination_angAcc = 0
        
        elif (self.inclination_angVel + self.inclination_angAcc) >= cfg.MAX_ANGVEL:
            self.inclination_angVel = cfg.MAX_ANGVEL
            self.inclination_angAcc = 0

        # Update heading
        self.azimuth_heading = (self.azimuth_heading + self.azimuth_angVel) % (cfg.MAX_AZIMUTH_ANGLE)
        
        if ((self.inclination_heading + self.inclination_angVel) < cfg.MAX_INCL_ANGLE) and ((self.inclination_heading + self.inclination_angVel) > cfg.MIN_INCL_ANGLE):
            self.inclination_heading= self.inclination_heading + self.inclination_angVel

        elif ((self.inclination_heading + self.inclination_angVel) >= cfg.MAX_INCL_ANGLE):
            self.inclination_heading = cfg.MAX_INCL_ANGLE
            self.inclination_angVel = 0
            self.inclination_angAcc = 0

        elif ((self.inclination_heading + self.inclination_angVel) <= cfg.MIN_INCL_ANGLE):
            self.inclination_heading = cfg.MIN_INCL_ANGLE
            self.inclination_angVel = 0
            self.inclination_angAcc = 0

        # Update position
        self.bitLocation.x += cfg.DRILL_SPEED * np.sin(self.inclination_heading)*np.cos(self.azimuth_heading)
        self.bitLocation.y += cfg.DRILL_SPEED *np.sin(self.inclination_heading)*np.sin(self.azimuth_heading)
        self.bitLocation.z += cfg.DRILL_SPEED * np.cos(self.inclination_heading)

        self.step_history.append([self.bitLocation.x,self.bitLocation.y, self.bitLocation.z])
=======
        if action == 0 and self.angAcc > -cfg.MAX_ANGACC:
            self.angAcc -= cfg.ANGACC_INCREMENT
        elif action == 1 and self.angAcc < cfg.MAX_ANGACC:
            self.angAcc += cfg.ANGACC_INCREMENT

        # Update angular velocity, if within limits
        if abs(self.angVel + self.angAcc) < cfg.MAX_ANGVEL:
            self.angVel += self.angAcc

        elif (self.angVel + self.angAcc) <= -cfg.MAX_ANGVEL:
            self.angVel = -cfg.MAX_ANGVEL
            self.angAcc = 0

        elif (self.angVel + self.angAcc) >= cfg.MAX_ANGVEL:
            self.angVel = cfg.MAX_ANGVEL
            self.angAcc = 0

        # Update heading.
        self.heading = (self.heading + self.angVel) % (2 * np.pi)

        # Update position
        self.bitLocation.x += cfg.DRILL_SPEED * np.sin(self.heading)
        self.bitLocation.y += cfg.DRILL_SPEED * np.cos(self.heading)
        self.step_history.append([self.bitLocation.x,self.bitLocation.y])
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py

    # Returns tuple of current state
    def get_state(self):
        # Core bit data
        state_list = [self.azimuth_heading,self.inclination_heading, self.azimuth_angVel,self.inclination_angVel, self.azimuth_angAcc,self.inclination_angAcc]
        # Target data that are inside the window
        for target in self.observation_space_container.target_window:

            state_list.append(target.center.z-self.bitLocation.z)
            state_list.append(es.get_horizontal_dist(self.bitLocation,target))
            state_list.append(es.get_relative_azimuth_angle(self.bitLocation, self.azimuth_heading,target))
            state_list.append(target.radius)
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py


        # Get all hazards inside window
        for hazard in self.observation_space_container.hazard_window:

            state_list.append(hazard.center.z-self.bitLocation.z)
            state_list.append(es.get_horizontal_dist(self.bitLocation,hazard))
            state_list.append(es.get_relative_azimuth_angle(self.bitLocation, self.azimuth_heading,hazard))
=======
        # Get hazards inside window
        for hazard in self.observation_space_container.hazard_window:
            state_list.append(hazard.center.x)
            state_list.append(hazard.center.y)
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
            state_list.append(hazard.radius)


        return tuple(state_list) 
            

    def reset(self):
        # Save previous run to log
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        if self.activate_log:
            self.write_to_log()
            self.episode_counter += 1
            self.total_reward = 0
=======
        #self.write_to_log()
        #self.episode_counter += 1
        #self.total_reward = 0
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
        
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y
        self.bitLocation.z = self.start_z

        self.azimuth_heading = uniform(0,2*np.pi)
        self.inclination_heading = uniform(0,np.pi/4)

        self.azimuth_angVel = self.initial_azimuth_angVel
        self.inclination_angVel = self.initial_inclination_angVel

<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        self.azimuth_angAcc = self.initial_azimuth_angAcc
        self.inclination_angAcc = self.initial_inclination_angAcc
        

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y,self.start_z]]       

        self.create_targets_and_hazards()

        # Re-configure the observation space
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,self.targets,self.hazards,self.bitLocation)
=======
        self.create_targets_and_hazards()

        # Re-configure the observation space
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,cfg.EXTRA_DATA_BOUNDS,self.targets,self.hazards,self.bitLocation)
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
      
        self.observation_space = self.observation_space_container.get_space_box()        
        
        self.state = self.get_state()
        
        return np.array(self.state)
    
    
    def write_to_log(self,*,filename="drill_log.txt"):
        f = open(filename,"a")
        text = "Episode nr: " +str(self.episode_counter) + " lasted for " + str(len(self.step_history)) + " steps. My total reward was: " + str(self.total_reward)  +"\n"
        
        f.write(text)
        f.close()
        #print("Log updated!")
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
    
=======
    """
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
   
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def display_state(self):
        print("Bit location (Not a part of ObsSpace):", Coordinate(self.bitLocation.x,self.bitLocation.y,self.bitLocation.z))
        print("Bit angles: ", self.state[0:2],"\n","angular velocities: ", self.state[2:4],"\n","angular accelerations: ", self.state[4:6])
        print("Targets inside window: ")
        for i in range(len(self.observation_space_container.target_window)):
            t = TargetBall(self.state[6+4*i],self.state[7+4*i],self.state[8+4*i],self.state[9+4*i])
            print(t)
        print("Hazards inside window")
        for i in range(len(self.observation_space_container.hazard_window)):
            h = Hazard(self.state[6+cfg.TARGET_WINDOW_SIZE*4+4*i],self.state[7+cfg.TARGET_WINDOW_SIZE*4+4*i],self.state[8+cfg.TARGET_WINDOW_SIZE*4+4*i],self.state[9+cfg.TARGET_WINDOW_SIZE*4+4*i])
            print(h)

    def get_xy_plane_figure(self):
        fig = plt.figure()

<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
=======
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
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
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
        axes.set_xlim(0,cfg.SCREEN_X)
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        axes.set_ylim(cfg.SCREEN_Y,0)
=======
        axes.set_ylim(0,cfg.SCREEN_Y)
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py

        plt.plot(x_positions,y_positions,"grey")
        plt.title("Well trajectory path in horizontal plane (x,y)")
        plt.legend()
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
        
        return fig

    def get_xz_plane_figure(self):
        fig1 = plt.figure()
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
        axes.set_xlim(0,cfg.SCREEN_X)
        axes.set_ylim(cfg.SCREEN_Z,0)

        plt.plot(x_positions,z_positions,"grey")
        plt.title("Well trajectory path in vertical plane (x,z)")
        plt.legend()
        
        return fig1    

    def get_3d_figure(self):
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
        #ax.inINCL_zaxis()

        # Plot circles from targetballs, colors just to verify the order of the balls
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
            plot_sphere(target.center.x,target.center.y,target.center.z,target.radius,"green",ax,str(cnt))
            #label = "Target #" + str(cnt)
            
            cnt += 1

        for hazard in self.hazards:
            plot_sphere(hazard.center.x,hazard.center.y,hazard.center.z,hazard.radius,"red",ax,'')

        # Set axis 
        #ax = plt.gca()
        ax.set_xlim(cfg.SCREEN_X,0)
        ax.set_ylim(0,cfg.SCREEN_Y)
        ax.set_zlim(cfg.SCREEN_Z,0)
        
        ax.set_xlabel("North")
        ax.set_ylabel("East")
        ax.set_zlabel("Down")

        return fig   

def plot_sphere(x0, y0, z0, r, c, ax, name=''):
    # Make data
    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 8)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_wireframe(x, y, z, color=c)
    ax.text(x0 + r, y0 + r, z0 + r, name, None)



def plot_ball(x0,y0,z0,r,c,ax, name):    
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))
    # Plot the surface
    ax.plot_surface(x, y, z, color=c)
    ax.text(x0+ r, y0 + r, z0 + r, name, None)
=======
        plt.show()
    

    def load_predefined_env(self,targets,hazards):
        self.targets = targets
        if self.activate_hazards:
            self.hazards = hazards
        else:
            self.hazards = []
        
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,cfg.EXTRA_DATA_BOUNDS,self.targets,self.hazards)
        self.observation_space = self.observation_space_container.get_space_box()        

        self.seed()

        self.state = self.get_state()

    def get_path(self):
        return self.step_history
       
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py

if __name__ == '__main__':
    startpos = Coordinate(100,900)
    """
    print("Testing init of targets and hazards")    
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
    startpos = Coordinate(100,100,100)

    print("Creating targets")
    t = es._init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.TARGET_RADII_BOUND,startpos)
=======

    print("Creating targets")
    t = es._init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_RADII_BOUND,startpos)
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
    for _ in t:
        print(_)
    
    print("Creating Hazards")    
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
    h = es._init_hazards(cfg.NUM_HAZARDS,cfg.HAZARD_BOUND_X,cfg.HAZARD_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.HAZARD_RADII_BOUND,startpos,t)
=======
    h = es._init_hazards(cfg.NUM_HAZARDS,cfg.HAZARD_BOUND_X,cfg.HAZARD_BOUND_Y,cfg.HAZARD_RADII_BOUND,startpos,t)
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
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
    axes.set_xlim(0,cfg.SCREEN_X)
    axes.set_ylim(0,cfg.SCREEN_Y)
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py

    
    plt.title("Test random generated hazard and targets")
    plt.show()
    
    
    print("Verify Environemnt")
    import random
    BIT_INITIALIZATION = [3.5*np.pi/4,np.pi/2, 0.0, 0.0, 0.0, 0.0]
=======
    
    plt.title("Test random generated hazard and targets")
    plt.show()
    """
    print("Verify Environemnt")
    import random
    BIT_INITIALIZATION = [np.pi/2,0.0,0.0]
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py

    env = DrillEnv(startpos,BIT_INITIALIZATION,activate_hazards=True)

    action_size = env.action_space.n
    action = random.choice(range(action_size))
    env.step(action)
    print("I took one step, this is what the current state is:")
<<<<<<< HEAD:gym-drill/gym_drill/envs/drill_env.py
    #print(env.state)
    print("\n display state: \n")
    env.display_state()
    print("\n") 
    print("\n display targets: \n")
    env.observation_space_container.display_targets()
    print("\n") 
    print("\n display hazards: \n")
    env.observation_space_container.display_hazards()
    print("\n") 
    env.display_3d_environment()

    for _ in range (500):
        action = random.choice(range(action_size))
        env.step(action)
    print("500 steps later")
    print("\n display state: \n")
    env.display_state()

    #env.observation_space_container.display_targets()
    #env.observation_space_container.display_hazards()
    env.display_3d_environment()

    print("\n Resetting \n")
    env.reset()
    print("\n display state: \n")
    env.display_state()
    #env.observation_space_container.display_targets()
    #env.observation_space_container.display_hazards()
    env.display_3d_environment()
   
=======
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
>>>>>>> 2914a7ab17b685790fbbcc82ad796d416a700bb1:2D-version/gym-drill/gym_drill/envs/drill_env.py
