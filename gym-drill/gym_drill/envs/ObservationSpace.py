from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Target import TargetBall
import numpy as np
import gym
from gym import spaces

SCREEN_X = 600
SCREEN_Y = 600

TARGET_WINDOW_SIZE = 2

# Used in space_box3. If needed they should be plart of arguments to init function
TARGET_BOUND_X = [0.5*SCREEN_X,0.9*SCREEN_X]
TARGET_BOUND_Y = [0.1*SCREEN_Y,0.6*SCREEN_Y]
TARGET_RADII_BOUND = [20,50]

class ObservationSpace:
    def __init__(self,space_bounds,bit_bounds,targets):
        # Spacial
        self.lower_x = space_bounds[0]
        self.upper_x = space_bounds[1]
        self.lower_y = space_bounds[2]
        self.upper_y = space_bounds[3] 

        # Bit related
        self.lower_heading = bit_bounds[0]
        self.upper_heading = bit_bounds[1]
        self.lower_ang_vel = bit_bounds[2]
        self.upper_ang_vel = bit_bounds[3]
        self.lower_ang_acc = bit_bounds[4]
        self.upper_ang_acc = bit_bounds[5]

        # Target related
        self.target_window = targets[:TARGET_WINDOW_SIZE]
        self.remaining_targets = targets[1:]
    
    # To print the obs_space. Can be nice for debugging purposes
    def __str__(self):
        text = "The observation space looks like this: \n \n" + "Spacial bounds: " \
        + str(self.lower_x) + " < x < " + str(self.upper_x) +" and " + str(self.lower_y) \
        + " < y < " + str(self.upper_y)+ ". \n \n" \
        + "Bit related bounds: \n" + "Heading interval: [" + str(self.lower_heading)+","\
        + str(self.upper_heading) + "] \n" \
        + "Angular velocity interval [" + str(self.lower_ang_vel) + ","+ str(self.upper_ang_vel) + "] \n"   \
        + "Angular acceleration interval [" + str(self.lower_ang_acc) + "," + str(self.upper_ang_acc) + "] \n \n" \
        + "There are " + str(TARGET_WINDOW_SIZE) + " targets inside the window. These are: \n" + str(self.target_window) + "\n \n" \
        + "The remaining targets are:" + str(self.remaining_targets)
    
        return text       
            

    # Shifts window. The last target will be loaded 3 times (fill the entire window)
    # When there are no more remaining_targets, nothing will happen 
    def shift_window(self):
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

    # Returns the environment
    def get_space_box0(self):
        lower = np.array([self.lower_x,self.lower_y,self.lower_heading,self.lower_ang_vel,self.lower_ang_acc])
        upper = np.array([self.upper_x,self.upper_y,self.upper_heading,self.upper_ang_vel,self.upper_ang_acc])

        # We append the targets as a list with targetball types (like an actual window)
        lower = np.append(lower,self.target_window)
        upper = np.append(upper,self.target_window)
        
        return spaces.Box(lower,upper,dtype=np.float64)   
    
    def get_space_box1(self):
        lower = np.array([self.lower_x,self.lower_y,self.lower_heading,self.lower_ang_vel,self.lower_ang_acc])
        upper = np.array([self.upper_x,self.upper_y,self.upper_heading,self.upper_ang_vel,self.upper_ang_acc])

        # We append the targets as type TargetBall to the observation space. Lower == Upper when it comes to targets
        for target in self.target_window:
            lower = np.append(lower,target)
            upper = np.append(upper,target)    
        
        return spaces.Box(lower,upper,dtype=np.float64)                    
    
    def get_space_box2(self):
        lower = np.array([self.lower_x,self.lower_y,self.lower_heading,self.lower_ang_vel,self.lower_ang_acc])
        upper = np.array([self.upper_x,self.upper_y,self.upper_heading,self.upper_ang_vel,self.upper_ang_acc])

        # We append the actual int values of the targets position and radius
        for t in self.target_window:
            lower = np.append(lower,[t.center.x,t.center.y,t.radius])
            lower = np.append(upper,[t.center.x,t.center.y,t.radius])

        return spaces.Box(lower,upper,dtype=np.float64)        
    
    def get_space_box3(self):
        lower = np.array([self.lower_x,self.lower_y,self.lower_heading,self.lower_ang_vel,self.lower_ang_acc])
        upper = np.array([self.upper_x,self.upper_y,self.upper_heading,self.upper_ang_vel,self.upper_ang_acc])

        # We append the upper and lower boundaries of where the target can exist. This is like we did before
        # This is the only case where we have an interval
        for t in self.target_window:
            lower = np.append(lower,[TARGET_BOUND_X[0],TARGET_BOUND_Y[0],TARGET_RADII_BOUND[0]])
            upper = np.append(upper,[TARGET_BOUND_X[1],TARGET_BOUND_Y[1],TARGET_RADII_BOUND[1]])

        print("Lower bounds have length: ",str(len(lower))," and look like this")
        print(lower)
        print("Upper bounds have length: ",str(len(upper))," and look like this:")
        print(upper)

        return spaces.Box(lower,upper,dtype=np.float64)    



"""
Possible problems:
    1. Not compatible with open AI gym or other packages 
       that might expect something specific when looking for the
       "observation_space variable". However I dont think so (we could probably
       rename the observations_space name to potato and it still works and in that
       case, there cannot be anything that looks for the name "observation_space")
       After some googling I now think that it needs to be called observation_space
"""