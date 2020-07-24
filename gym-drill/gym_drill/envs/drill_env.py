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
from gym_drill.envs import rdwellpath as rwp
from gym_drill.envs import environment_config as cfg

class DrillEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
}

    def __init__(self,startLocation,bitInitialization,*,activate_hazards=False,monte_carlo=False):
        self.activate_hazards = activate_hazards
        self.monte_carlo = monte_carlo

        self.start_x = startLocation.x
        self.start_y = startLocation.y
        self.start_z = startLocation.z
        # Save the starting position as "first" step. Needed for plotting in matplotlib
        self.step_history = [[self.start_x,self.start_y,self.start_z]]        

        # We init parameters here        
        self.bitLocation = startLocation
        self.horizontal_heading = uniform(0,np.pi/2)
        self.vertical_heading = uniform(0,np.pi/4)

        #self.angVel = bitInitialization[1]
        self.horizontal_angVel = bitInitialization[2]
        self.vertical_angVel =bitInitialization[3]

        #self.angAcc = bitInitialization[2]
        self.horizontal_angAcc = bitInitialization[4]
        self.vertical_angAcc =bitInitialization[5]      

        # For resetting the environment        
        self.initial_horizontal_heading = bitInitialization[0]
        self.initial_vertical_heading = bitInitialization[1]        
        self.initial_horizontal_angVel = bitInitialization[2]
        self.initial_vertical_angVel = bitInitialization[3]
        self.initial_horizontal_angAcc = bitInitialization[4]
        self.initial_vertical_angAcc = bitInitialization[5]
        
        # Generate feasible environments to train in using a Monte Carlo simulation 
        if self.monte_carlo:
            print("Running", str(cfg.NUM_MONTE_CARLO_ENVS),"Monte Carlo simulations to generate target sets!")
            rwp.random_targetset_to_file(cfg.ENVIRONMENT_FILENAME,cfg.NUM_MONTE_CARLO_ENVS,cfg.NUM_TARGETS,[self.bitLocation.x,self.bitLocation.y,self.bitLocation.z],120,200)      

        self.create_targets_and_hazards()
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,self.targets,self.hazards,self.bitLocation)
        self.observation_space = self.observation_space_container.get_space_box()        
        self.action_space = spaces.Discrete(9)     

        self.seed()
        self.viewer = None
        self.state = self.get_state()

        # Log related
        """
        self.episode_counter = 0 # Used to write to log
        self.total_reward = 0      
        es._init_log()
        """
    def create_targets_and_hazards(self):
        if not self.monte_carlo:
            self.targets = es._init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.TARGET_RADII_BOUND,self.bitLocation)
            if self.activate_hazards:
                self.hazards = es._init_hazards(cfg.NUM_HAZARDS,cfg.HAZARD_BOUND_X,cfg.HAZARD_BOUND_Y,cfg.HAZARD_BOUND_Z,cfg.HAZARD_RADII_BOUND,self.bitLocation,self.targets)
            else:
                self.hazards = []
        else:
            linenr = np.random.randint(1,cfg.NUM_MONTE_CARLO_ENVS)
            self.targets,self.hazards = es._read_env_from_file(cfg.ENVIRONMENT_FILENAME,linenr)
            # Overwrite hazards if not activated
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
        #self.total_reward += reward
        return np.array(self.state), reward, done, {}

    
    # Returns the reward for the step and if episode is over
    def get_reward_and_done_signal(self):
        done = False      
        reward = cfg.STEP_PENALTY
        
        if self.horizontal_angAcc != 0:
            reward += cfg.ANGULAR_ACCELERATION_PENALTY
        
        if self.horizontal_angVel != 0:
            reward += cfg.ANGULAR_VELOCITY_PENALTY
        
        if self.vertical_angAcc != 0:
            reward += cfg.ANGULAR_ACCELERATION_PENALTY
        
        if self.vertical_angVel != 0:
            reward += cfg.ANGULAR_VELOCITY_PENALTY
        

        # If drill is no longer on screen, game over.
        if not (0 < self.bitLocation.x < cfg.SCREEN_X and 0 < self.bitLocation.y < cfg.SCREEN_Y and 0 < self.bitLocation.z < cfg.SCREEN_Z):
            reward  += cfg.OUTSIDE_SCREEN_PENALTY
            done = True   
        
        # Check if we hit a hazard
        for h in self.observation_space_container.hazard_window:
            if es._is_within(self.bitLocation,h.center,h.radius) and not h.is_hit:
                reward -= cfg.HAZARD_PENALTY
                h.is_hit = True
                #done = True

        if len(self.step_history)>cfg.NUM_MAX_STEPS:
            done= True                        

        # Find the values of the current target
        current_target = self.observation_space_container.target_window[0] #object of target class
        current_target_pos = np.array([current_target.center.x, current_target.center.y, current_target.center.z])
        current_target_rad = current_target.radius
        drill_pos = np.array([self.bitLocation.x, self.bitLocation.y, self.bitLocation.z])

        # Check if target is hit
        if np.linalg.norm(current_target_pos - drill_pos) < current_target_rad:
        #if es._is_within(self.bitLocation,self.observation_space_container.target_window[0].center,self.observation_space_container.target_window[0].radius):
            reward += cfg.TARGET_REWARD

            if len(self.observation_space_container.remaining_targets) == 0:
                reward += (cfg.NUM_MAX_STEPS-len(self.step_history))*cfg.FINISHED_EARLY_FACTOR
                done = True
            else:
                self.observation_space_container.shift_target_window()
        
        else:
            reward_factor = np.cos(self.get_relative_horizontal_angle(current_target)) # value between -1 and +1 
            reward += reward_factor*cfg.ANGLE_REWARD_FACTOR #reward for having a correct horizontal-angle
            height_diff = current_target_pos[2]-self.bitLocation.z
            if height_diff != 0:
                reward += np.cos(self.vertical_heading)*(height_diff/abs(height_diff))*0.5 #should be global constant #reward for going in the right directen (up/down)
        return reward, done


    def get_xy_dist(self,obj):
        return np.sqrt((obj.center.x -self.bitLocation.x)**2+(obj.center.y - self.bitLocation.y)**2)
        
    def get_relative_horizontal_angle(self,obj):
        object_hor_pos_vector = np.array([obj.center.x,obj.center.y])
        curr_drill_hor_pos_vector = np.array([self.bitLocation.x,self.bitLocation.y])
        appr_vec = object_hor_pos_vector - curr_drill_hor_pos_vector
        head_vec = np.array([np.cos(self.horizontal_heading), np.sin(self.horizontal_heading)])
        angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))

        return angle_between_vectors
    """
    def get_horizontal_angle_relative_to_target(self):
        current_target = self.observation_space_container.target_window[0]
                
        curr_target_hor_pos_vector = np.array([current_target.center.x,current_target.center.y])

        curr_drill_hor_pos_vector = np.array([self.bitLocation.x,self.bitLocation.y])
        appr_vec = curr_target_hor_pos_vector - curr_drill_hor_pos_vector

        head_vec = np.array([np.cos(self.horizontal_heading), np.sin(self.horizontal_heading)])
        angle_between_vectors = np.math.atan2(np.linalg.det([appr_vec, head_vec]), np.dot(appr_vec, head_vec))

        return angle_between_vectors
    """
    # For encapsulation. Updates the bit according to the action
    def update_bit(self,action):
        # Update angular acceleration, if within limits
        if action < 3 and self.vertical_angAcc < cfg.MAX_ANGACC:            #indexes of action space:
            self.vertical_angAcc += cfg.ANGACC_INCREMENT                    #   0       1       2 | (0-2): accelerate upwards
        elif action > 5 and self.vertical_angAcc > -cfg.MAX_ANGACC:         #   3       4       5 | (3-5): don't accelate in the vertical plane
            self.vertical_angAcc -= cfg.ANGACC_INCREMENT                    #   6       7       8 | (6-8): accelerate downwards
                                                                        #---------------------
                                                                        #(0,3,6): accelerate left 
                                                                        #        (1,4,7): don't accelerate in the horizontal plane
                                                                        #                (2,5,8): accelerate right
        if (action == 0 or action == 3 or action == 6) and self.horizontal_angAcc > -cfg.MAX_ANGACC:
            self.horizontal_angAcc -= cfg.ANGACC_INCREMENT
        elif (action == 2 or action == 5 or action == 8) and self.horizontal_angAcc < cfg.MAX_ANGACC:
            self.horizontal_angAcc += cfg.ANGACC_INCREMENT
        

        # Update angular velocity

            # horizontal
        if abs(self.horizontal_angVel + self.horizontal_angAcc) < cfg.MAX_ANGVEL:
            self.horizontal_angVel += self.horizontal_angAcc

        elif (self.horizontal_angVel + self.horizontal_angAcc) <= -cfg.MAX_ANGVEL:
            self.horizontal_angVel = -cfg.MAX_ANGVEL
            self.horizontal_angAcc = 0
        
        elif (self.horizontal_angVel + self.horizontal_angAcc) >= cfg.MAX_ANGVEL:
            self.horizontal_angVel = cfg.MAX_ANGVEL
            self.horizontal_angAcc = 0

            # vertical
        if abs(self.vertical_angVel + self.vertical_angAcc) < cfg.MAX_ANGVEL:
            self.vertical_angVel += self.vertical_angAcc

        elif (self.vertical_angVel + self.vertical_angAcc) <= -cfg.MAX_ANGVEL:
            self.vertical_angVel = -cfg.MAX_ANGVEL
            self.vertical_angAcc = 0
        
        elif (self.vertical_angVel + self.vertical_angAcc) >= cfg.MAX_ANGVEL:
            self.vertical_angVel = cfg.MAX_ANGVEL
            self.vertical_angAcc = 0


        # Update heading.

        self.horizontal_heading = (self.horizontal_heading + self.horizontal_angVel) % (2 * np.pi)

        
        if ((self.vertical_heading + self.vertical_angVel) < cfg.MAX_VERT_ANGLE) and ((self.vertical_heading + self.vertical_angVel) > cfg.MIN_VERT_ANGLE):
            self.vertical_heading= self.vertical_heading + self.vertical_angVel

        elif ((self.vertical_heading + self.vertical_angVel) >= cfg.MAX_VERT_ANGLE):
            self.vertical_heading = cfg.MAX_VERT_ANGLE
            self.vertical_angVel = 0
            self.vertical_angAcc = 0

        elif ((self.vertical_heading + self.vertical_angVel) <= cfg.MIN_VERT_ANGLE):
            self.vertical_heading = cfg.MIN_VERT_ANGLE
            self.vertical_angVel = 0
            self.vertical_angAcc = 0



        # Update position
        self.bitLocation.x += cfg.DRILL_SPEED * np.sin(self.vertical_heading)*np.cos(self.horizontal_heading)
        self.bitLocation.y += cfg.DRILL_SPEED *np.sin(self.vertical_heading)*np.sin(self.horizontal_heading)
        self.bitLocation.z += cfg.DRILL_SPEED * np.cos(self.vertical_heading)

        self.step_history.append([self.bitLocation.x,self.bitLocation.y, self.bitLocation.z])

    # Returns tuple of current state
    def get_state(self):
        # Core bit data
        state_list = [self.horizontal_heading,self.vertical_heading, self.horizontal_angVel,self.vertical_angVel, self.horizontal_angAcc,self.vertical_angAcc]
        # Target data that are inside the window
        for target in self.observation_space_container.target_window: # This will cause bug
            """
            state_list.append(target.center.x)
            state_list.append(target.center.y)
            state_list.append(target.center.z)
            state_list.append(target.radius)
            """
            state_list.append(target.center.z-self.bitLocation.z)
            state_list.append(self.get_xy_dist(target))
            state_list.append(self.get_relative_horizontal_angle(target))
            state_list.append(target.radius)


        # Get all hazards inside window
        for hazard in self.observation_space_container.hazard_window:
            """
            state_list.append(hazard.center.x)
            state_list.append(hazard.center.y)
            state_list.append(hazard.center.z)
            state_list.append(hazard.radius)
            """
            state_list.append(hazard.center.z-self.bitLocation.z)
            state_list.append(self.get_xy_dist(hazard))
            state_list.append(self.get_relative_horizontal_angle(hazard))
            state_list.append(hazard.radius)

        # Extra data
        """
        current_target = self.observation_space_container.target_window[0]
        height_diff = current_target.center.z - self.bitLocation.z

        relative_angle = self.get_horizontal_angle_relative_to_target()

        state_list =  state_list + [height_diff, relative_angle]
        """
        return tuple(state_list) 
            

    def reset(self):
        # Save previous run to log
        #self.write_to_log()
        #self.episode_counter += 1
        self.total_reward = 0
        
        self.bitLocation.x = self.start_x
        self.bitLocation.y = self.start_y
        self.bitLocation.z = self.start_z

        self.horizontal_heading = uniform(0,2*np.pi)
        self.vertical_heading = uniform(0,np.pi/4)

        self.horizontal_angVel = self.initial_horizontal_angVel
        self.vertical_angVel = self.initial_vertical_angVel

        self.horizontal_angAcc = self.initial_horizontal_angAcc
        self.vertical_angAcc = self.initial_vertical_angAcc
        

        # Save the starting position as "first" step
        self.step_history = [[self.start_x,self.start_y,self.start_z]]       

        self.create_targets_and_hazards()

        # Re-configure the observation space
        self.observation_space_container= ObservationSpace(cfg.SPACE_BOUNDS,cfg.TARGET_BOUNDS,cfg.HAZARD_BOUNDS,cfg.BIT_BOUNDS,self.targets,self.hazards,self.bitLocation)
      
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
        print("Bit location:", Coordinate(self.bitLocation.x,self.bitLocation.y,self.bitLocation.z))
        print("Bit angles: ", self.state[0:5])
        print("Targets inside window: ")
        for i in range(len(self.observation_space_container.target_window)):
            t = TargetBall(self.state[6+4*i],self.state[7+4*i],self.state[8+4*i],self.state[9+4*i])
            print(t)
        print("Hazards inside window")
        for i in range(len(self.observation_space_container.hazard_window)):
            h = Hazard(self.state[6+cfg.TARGET_WINDOW_SIZE*4+4*i],self.state[7+cfg.TARGET_WINDOW_SIZE*4+4*i],self.state[8+cfg.TARGET_WINDOW_SIZE*4+4*i],self.state[9+cfg.TARGET_WINDOW_SIZE*4+4*i])
            print(h)
        """
        print("Extra data:")
        print("Distance: ",self.state[9+4*len(self.observation_space_container.target_window)+ 4*len(self.observation_space_container.hazard_window)])
        print("Relative angle: ",self.state[9+4*len(self.observation_space_container.target_window)+ 4*len(self.observation_space_container.hazard_window)+1])
        """

    def display_planes(self):
        plt.subplot(2,1,1)
    #def display_horizontal_plane_of_environment(self):
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
        axes.set_xlim(0,cfg.SCREEN_X)
        axes.set_ylim(cfg.SCREEN_Y,0)

        plt.plot(x_positions,y_positions,"grey")
        plt.title("Well trajectory path in horizontal plane (x,y)")
        plt.legend()
        #plt.show()
    
    #def display_vertical_plane_of_environment(self):
        # Get data
        plt.subplot(2,1,2)
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
        
        plt.show()
    

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

            plot_ball(target.center.x,target.center.y,target.center.z,target.radius,colors_order[cnt],ax,str(cnt))
            #label = "Target #" + str(cnt)
            
            cnt += 1

        for hazard in self.hazards:
            plot_ball(hazard.center.x,hazard.center.y,hazard.center.z,hazard.radius,'k',ax,'')

        # Set axis 
        #ax = plt.gca()
        ax.set_xlim(cfg.SCREEN_X,0)
        ax.set_ylim(0,cfg.SCREEN_Y)
        ax.set_zlim(cfg.SCREEN_Z,0)
        
        ax.set_xlabel("North")
        ax.set_ylabel("East")
        ax.set_zlabel("Down")

        plt.show()

   
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

if __name__ == '__main__':
    print("Testing init of targets and hazards")    
    startpos = Coordinate(100,100,100)

    print("Creating targets")
    t = es._init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.TARGET_RADII_BOUND,startpos)
    for _ in t:
        print(_)
    
    print("Creating Hazards")    
    h = es._init_hazards(cfg.NUM_HAZARDS,cfg.HAZARD_BOUND_X,cfg.HAZARD_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.HAZARD_RADII_BOUND,startpos,t)
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

    
    plt.title("Test random generated hazard and targets")
    plt.show()
    
    
    print("Verify Environemnt")
    import random
    BIT_INITIALIZATION = [3.5*np.pi/4,np.pi/2, 0.0, 0.0, 0.0, 0.0]

    env = DrillEnv(startpos,BIT_INITIALIZATION,activate_hazards=True)

    action_size = env.action_space.n
    action = random.choice(range(action_size))
    env.step(action)
    print("I took one step, this is what the current state is:")
    print(env.state)

    env.observation_space_container.display_hazards()
    env.display_3d_environment()

    for _ in range (5000):
        action = random.choice(range(action_size))
        env.step(action)
    print("50 steps later")
    env.display_state()
    env.observation_space_container.display_hazards()
    env.display_3d_environment()

    print("Resetting")
    env.reset()
    env.display_state()
    env.observation_space_container.display_hazards()
    env.display_3d_environment()
   