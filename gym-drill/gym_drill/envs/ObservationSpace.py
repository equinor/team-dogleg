from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.Hazard import Hazard
from gym_drill.envs import environment_support as es

import numpy as np
import gym
from gym import spaces

# Designited slots in the observation space
TARGET_WINDOW_SIZE = 3
HAZARD_WINDOW_SIZE = 0 # MUST HAVE AT LEAST THIS MANY HAZARDS

# Targets are assumed to be ordered
class ObservationSpace:
    def __init__(self,space_bounds,target_bounds,hazard_bounds,bit_bounds,extra_data,targets,hazards, bit_starting_pos):
        # Spacial
        self.lower_x = space_bounds[0]
        self.upper_x = space_bounds[1]
        self.lower_y = space_bounds[2]
        self.upper_y = space_bounds[3]
        self.lower_z = space_bounds[4]
        self.upper_z = space_bounds[5]

        # Bit related
        self.lower_horizontal_heading = bit_bounds[0]
        self.upper_horizontal_heading = bit_bounds[1]
        self.lower_vertical_heading = bit_bounds[2]
        self.upper_vertical_heading = bit_bounds[3]
        self.lower_ang_vel = bit_bounds[4]
        self.upper_ang_vel = bit_bounds[5]
        self.lower_ang_acc = bit_bounds[6]
        self.upper_ang_acc = bit_bounds[7]

        # Target related
        self.target_window = targets[:TARGET_WINDOW_SIZE]
        self.remaining_targets = targets[1:]
        self.target_bound_x = target_bounds[0]
        self.target_bound_y = target_bounds[1]
        self.target_bound_z = target_bounds[2]
        self.target_bound_r = target_bounds[3]

        # Hazard related
        self.hazards = hazards
        self.hazard_window = self.find_closest_hazards(bit_starting_pos)
        self.hazard_bound_x = hazard_bounds[0]
        self.hazard_bound_y = hazard_bounds[1]
        self.hazard_bound_z = hazard_bounds[2]
        self.hazard_bound_r = hazard_bounds[3]

        # Extra data
        self.target_distance_bound = extra_data[0]
        self.relative_angle_bound = extra_data[1]

    def display_targets(self):
        print("The current target window looks like this:")
        for w in self.target_window:
            print(w)
        print("The remaining targets are:")
        for t in self.remaining_targets:
            print(t)

    def display_hazards(self):
        print("The current hazard window looks like this:")
        for w in self.hazard_window:
            print(w)
        print("All hazards are:")
        for h in self.hazards:
            print(h)  
            
    # To print the obs_space. Can be nice for debugging purposes
    def __str__(self):
        text = "The observation space looks like this: \n \n" + "Spacial bounds: " \
        + str(self.lower_x) + " < x < " + str(self.upper_x) +" and " + str(self.lower_y) \
        + " < y < " + str(self.upper_y)+ ". \n \n" \
        + "Bit related bounds: \n" + "Horisontal heading interval: [" + str(self.lower_horizontal_heading)+","\
        + str(self.upper_horizontal_heading) + "] \n" \
        + "Vertical heading interval: [" + str(self.lower_vertical_heading)+","\
        + str(self.upper_vertical_heading) + "] \n" \
        + "Angular velocity interval [" + str(self.lower_ang_vel) + ","+ str(self.upper_ang_vel) + "] \n"   \
        + "Angular acceleration interval [" + str(self.lower_ang_acc) + "," + str(self.upper_ang_acc) + "] \n \n" \
        + "There are " + str(TARGET_WINDOW_SIZE) + " targets inside the window. These are: \n" 
        
        for t in self.target_window:
            text = text + str(t) + "\n"
                
        text = text + "There are " + str(len(self.remaining_targets))+  " remaining targets. These are: \n" 
        for t in self.remaining_targets:
            text = text + str(t) + "\n"       
                
        text = text + "Target bounds are: \n" + "x: " + str(self.target_bound_x) + "\n" \
        + "y: " + str(self.target_bound_y) + "\n" \
        + "z: " + str(self.target_bound_z) + "\n" \
        + "r: " + str(self.target_bound_r) + "\n" \
        + "Hazard bounds are : \n" + "x: " + str(self.hazard_bound_x) + "\n" \
        + "y: " + str(self.hazard_bound_y) + "\n" \
        + "z: " + str(self.hazard_bound_z) + "\n" \
        + "r: " + str(self.hazard_bound_r) + "\n" \
        + "The hazards inside the window are: \n"
        for h in self.hazard_window:
            text = text + str(h) + "\n"
        text = text + "There are a total of" + str(len(self.hazards)) + " hazards, these are \n" 
        for h in self.hazards:
            text = text + str(h) + "\n"    

        text = text + "The extra data bounds are: \n" \
        + "Target distance: " + str(self.target_distance_bound) +"\n" \
        + "Relative angle " + str(self.relative_angle_bound)

        return text      
    
    def find_closest_hazards(self,bitPostion):
        # Need to make a independent copy that does not point to same memory location
        candidates = [] 
        for h in self.hazards:
            candidates.append(h)

        window = []
        for _ in range(HAZARD_WINDOW_SIZE):
            closest_index = es._findNearest(bitPostion,candidates)
            window.append(candidates[closest_index])
            candidates.pop(closest_index)

        return window


    def update_hazard_window(self,bitPosition):
        self.hazard_window = self.find_closest_hazards(bitPosition)

    # Shifts window. The last target will be loaded 3 times (fill the entire window)
    # When there are no more remaining_targets, nothing will happen 
    def shift_target_window(self):
        if len(self.remaining_targets) >= TARGET_WINDOW_SIZE:
            self.target_window = self.remaining_targets[:TARGET_WINDOW_SIZE]
            self.remaining_targets.pop(0)
        elif len(self.remaining_targets) > 0:
            # Load like normal
            self.target_window = self.remaining_targets[:TARGET_WINDOW_SIZE]
            self.remaining_targets.pop(0)
            while len(self.target_window) < TARGET_WINDOW_SIZE:
                # Add the last element of target_window, until window is big enough
                self.target_window.append(self.target_window[-1:][0])
        else:
            print("No more targets to add to window, we are done!")    
    
    def get_space_box(self):#This has to be updated to fit the 3D environment
        lower = np.array([self.lower_x,self.lower_y,self.lower_z,self.lower_horizontal_heading,self.lower_vertical_heading,self.lower_ang_vel,self.lower_ang_vel,self.lower_ang_acc,self.lower_ang_acc])
        upper = np.array([self.upper_x,self.upper_y,self.upper_z,self.upper_horizontal_heading,self.upper_vertical_heading,self.upper_ang_vel,self.upper_ang_vel,self.upper_ang_acc,self.upper_ang_acc])

        for _ in range(TARGET_WINDOW_SIZE):
            lower = np.append(lower,[self.target_bound_x[0],self.target_bound_y[0],self.target_bound_z[0],self.target_bound_r[0]])
            upper = np.append(upper,[self.target_bound_x[1],self.target_bound_y[1],self.target_bound_z[1],self.target_bound_r[1]])

        for _ in range(HAZARD_WINDOW_SIZE):
            lower = np.append(lower,[self.hazard_bound_x[0],self.hazard_bound_y[0],self.hazard_bound_z[0],self.hazard_bound_r[0]])
            upper = np.append(upper,[self.hazard_bound_x[1],self.hazard_bound_y[1],self.hazard_bound_z[1],self.hazard_bound_r[1]])       
        
        # Add extra data
        lower = np.append(lower,[self.target_distance_bound[0],self.relative_angle_bound[0]])
        upper = np.append(upper,[self.target_distance_bound[1],self.relative_angle_bound[1]])
        
        return spaces.Box(lower,upper,dtype=np.float64)    


if __name__ == '__main__':
    # Test basic functionality
    SCREEN_X = 2000
    SCREEN_Y = 2000
    SCREEN_Z = 2000

    SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y,0,SCREEN_Z] # x_low,x_high,y_low,y_high
    BIT_BOUNDS = [0,2*np.pi,0,np.pi,-0.05,0.05,-0.05,0.05,-0.1,0.1,-0.1,0.1] #

    TARGET_BOUND_X = [0.5*SCREEN_X,0.9*SCREEN_X]
    TARGET_BOUND_Y = [0.1*SCREEN_Y,0.6*SCREEN_Y]
    TARGET_BOUND_Z = [0.5*SCREEN_Z,0.9*SCREEN_Z]
    TARGET_RADII_BOUND = [20,50]
    TARGET_BOUNDS = [TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_BOUND_Z,TARGET_RADII_BOUND]

    HAZARD_BOUND_X = [0,SCREEN_X]
    HAZARD_BOUND_Y = [0,SCREEN_Y]
    HAZARD_BOUND_Z = [0,SCREEN_Z]
    HAZARD_RADII_BOUND = [20,50]
    HAZARD_BOUNDS = [HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_BOUND_Z,HAZARD_RADII_BOUND]
    
    targets = []
    for _ in range(4):
        target_center = Coordinate(np.random.uniform(TARGET_BOUND_X[0],TARGET_BOUND_X[1]),np.random.uniform(TARGET_BOUND_Y[0],TARGET_BOUND_Y[1]),np.random.uniform(TARGET_BOUND_Z[0],TARGET_BOUND_Z[1]))
        target_radius = np.random.uniform(TARGET_RADII_BOUND[0],TARGET_RADII_BOUND[1])
        target_candidate = TargetBall(target_center.x,target_center.y,target_center.z,target_radius)
        targets.append(target_candidate)
    
    hazards = []
    # Additional data
    DIAGONAL = np.sqrt(SCREEN_X**2 + SCREEN_Y**2 + SCREEN_Z**2)
    TARGET_DISTANCE_BOUND = [0,DIAGONAL]
    RELATIVE_ANGLE_BOUND = [-np.pi,np.pi]
    EXTRA_DATA_BOUNDS = [TARGET_DISTANCE_BOUND,RELATIVE_ANGLE_BOUND]

    
    for _ in range(4):
        hazard_center = Coordinate(np.random.uniform(HAZARD_BOUND_X[0],HAZARD_BOUND_X[1]),np.random.uniform(HAZARD_BOUND_Y[0],HAZARD_BOUND_Y[1]),np.random.uniform(HAZARD_BOUND_Z[0],HAZARD_BOUND_Z[1]))
        hazard_radius = np.random.uniform(HAZARD_RADII_BOUND[0],HAZARD_RADII_BOUND[1])
        hazard_candidate = Hazard(hazard_center.x,hazard_center.y,hazard_center.z,hazard_radius)
        hazards.append(hazard_candidate)
     

    print("Creating obs_space")
    print()
    obs_space = ObservationSpace(SPACE_BOUNDS,TARGET_BOUNDS,HAZARD_BOUNDS,BIT_BOUNDS,EXTRA_DATA_BOUNDS,targets,hazards,Coordinate(1000,1000,1000))
    #print(obs_space)
    print("test hazard window")
    obs_space.update_hazard_window(Coordinate(600,600,600))
    print(obs_space.hazard_window)
    
    
    box = obs_space.get_space_box()
    print(box)
    print("Expected dimension of the obs space is: ", 9 + 4*TARGET_WINDOW_SIZE + 4*(HAZARD_WINDOW_SIZE) + 2) # Only 2 extra data
    """
    print("Test shifting of window")
    print("State before shifting")
    obs_space.display_targets()
    print("-----------------------------------------------------------------------")
    print("First shift")
    obs_space.shift_target_window()
    obs_space.display_targets()
    print("-----------------------------------------------------------------------")
    print("Second shift")
    obs_space.shift_target_window()
    obs_space.display_targets()
    print("-----------------------------------------------------------------------")
    print("Third shift")
    obs_space.shift_target_window()
    obs_space.display_targets()
    print("-----------------------------------------------------------------------")
    print("Fourth shift")
    obs_space.shift_target_window()
    obs_space.display_targets()
    print("-----------------------------------------------------------------------")
    print("Fifth shift")
    obs_space.shift_target_window()
    obs_space.display_targets()

    print("im done")
    """





