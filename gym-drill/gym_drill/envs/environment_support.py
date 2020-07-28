"""
A place to implement smaller custom support functions to be used in the environment
"""
import numpy as np
from datetime import datetime
import sys

from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.Hazard import Hazard

def _init_log(*,filename="drill_log.txt"):
    print("Initiating log..")
    print("Running with log greatly reduces performance and is not recommended during longer training sessions!")
    print("It is recommended to run with log when developing i.e the reward system of the environment.")
    f = open(filename,"w")
    init_msg = "Log for training session started at " + str(datetime.now()) +"\n \n"
    f.write(init_msg)
    f.close()

# Returns position of Targets and Hazard as specified in filename
# Line number has zero indexing and line 0 cannot be used as it is the documentation line.
def _read_env_from_file(filename,line_number):
    if line_number == 0:
        print("Cannot read environments specification from line 0 as it is used for documentation")
        print("Line number must be 1 or higher!")
        sys.exit()

    targets = []
    hazards = []
    file = open(filename)
    environment_line = ""

    # We iterate to the desired line to avoid loading all lines into memory
    for i, line in enumerate(file):
        if i == line_number:
            environment_line = line
            break
    
    target_string = environment_line.split("/")[0]
    target_list_string = target_string.split(";")
    for t in target_list_string:
        # "10,10,5"
        l = t.split(",")
        try:
            x =l[0]
            y = l[1]
            z = l[2]
            r = l[3]
            x = int(x)
            y = int(y)
            z = int(z)
            r = int(r)

        except Exception:
            print(target_string)
            print(target_list_string)
            print("t;",t)
            print("l;",l)
        
            raise ValueError("Coordinates in file are not numbers!")
        target_ball = TargetBall(x,y,z,r)
        targets.append(target_ball)   
    
    try: 
        hazard_string = environment_line.split("/")[1]
        hazard_list_string = hazard_string.split(";")
        
        for h in hazard_list_string:
            # "10,10,5"
            l = h.split(",")
            try:
                x = l[0]
                y = l[1]
                z = l[2]
                r = l[3]
                x = int(x)
                y = int(y)
                z = int(z)
                r = int(r)
            except Exception:
                raise ValueError("Coordinates in file are not numbers!")
            
            hazard_ball = Hazard(x,y,z,r)
            hazards.append(hazard_ball)
    except Exception:
        hazards = []   
    
    return targets, hazards


# Returns an ordered list of randomly generated targets within the bounds given. 
def _init_targets(num_targets,x_bound,y_bound,z_bound,r_bound,start_location):
    all_targets = []

    for t in range(num_targets):
        target = _create_unique_random_target(start_location,x_bound,y_bound,z_bound,r_bound,all_targets)
        all_targets.append(target)        
    
    all_targets = _orderTargets(start_location,all_targets)

    return all_targets

def _init_hazards(num_hazards,x_bound,y_bound,z_bound,r_bound,start_pos,existing_targets):
    all_hazards = []
    for h in range(num_hazards):
        avoid = existing_targets + all_hazards
        hazard = _create_unique_random_hazard(start_pos,x_bound,y_bound, z_bound,r_bound,avoid)
        all_hazards.append(hazard)

    return all_hazards

# Finds nearest between 1 point and a list of candidate points
# startlocation is type Coordinate, and candidates is list of types Targets
# Also works with Hazards
def _findNearest(start_location,candidates):
    current_shortest_distance = -1 # Init with an impossible distance
    current_closest_target_index = 0
    for candidate_index in range(len(candidates)):        
        candidate = candidates[candidate_index]     
        distance = Coordinate.getEuclideanDistance(candidate.center,start_location)
        distance -= candidate.radius

        if distance < current_shortest_distance or current_shortest_distance == -1:           
            current_shortest_distance = distance
            current_closest_target_index = candidate_index
        
    return current_closest_target_index

# Orders the target based upon a given start location
# start_location is type Coordinate, all_targets is list of type targets
def _orderTargets(start_location,all_targets):
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
def _create_unique_random_target(start_pos,x_bound,y_bound, z_bound,r_bound,existing_targets):
    target_center = Coordinate(np.random.uniform(x_bound[0],x_bound[1]),(np.random.uniform(y_bound[0],y_bound[1])),(np.random.uniform(z_bound[0],z_bound[1])))
    target_radius = np.random.uniform(r_bound[0],r_bound[1])
    target_candidate = TargetBall(target_center.x,target_center.y,target_center.z,target_radius)

    for target in existing_targets:
        if _is_overlapping(target,target_candidate) or _is_within(start_pos,target_center,target_radius):
            target_candidate =_create_unique_random_target(start_pos,x_bound,y_bound,z_bound,r_bound,existing_targets)
            break

    return target_candidate

# Creates a uniqe hazard that does not overlad with any obstacles in existing_obstacles
def _create_unique_random_hazard(start_pos,x_bound,y_bound,z_bound,r_bound,existing_obstacles):
    hazard_center = Coordinate(np.random.uniform(x_bound[0],x_bound[1]),(np.random.uniform(y_bound[0],y_bound[1])),(np.random.uniform(z_bound[0],z_bound[1] )))
    hazard_radius = np.random.uniform(r_bound[0],r_bound[1])
    hazard_candidate = Hazard(hazard_center.x,hazard_center.y,hazard_center.z,hazard_radius)  
    
    for obstacle in existing_obstacles:
        if _is_overlapping(obstacle,hazard_candidate) or _is_within(start_pos,hazard_center,hazard_radius):
            hazard_candidate = _create_unique_random_hazard(start_pos,x_bound,y_bound,z_bound,r_bound,existing_obstacles)
            break
    
    return hazard_candidate

def _is_within(bitPosition,targetPosition,targetRadius):
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 +(bitPosition.z - targetPosition.z)**2 < targetRadius**2

def angle_between_vectors(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
if __name__ == '__main__':

    from gym_drill.envs import environment_config as cfg
    #targets, hazards = _read_env_from_file("environments.txt",1)
    #for t in targets:
    #    print(t)
    #for h in hazards:
    #    print(h)

    targets = _init_targets(cfg.NUM_TARGETS,cfg.TARGET_BOUND_X,cfg.TARGET_BOUND_Y,cfg.TARGET_BOUND_Z,cfg.TARGET_RADII_BOUND,Coordinate(100,100,900))
    for t in targets:
        print(t)