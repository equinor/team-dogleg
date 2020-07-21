"""
A place to implement smaller custom support functions to be used in the environment
"""
import numpy as np
from datetime import datetime

from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.Hazard import Hazard

def _init_log(*,filename="drill_log.txt"):
    f = open(filename,"w")
    init_msg = "Log for training session started at " + str(datetime.now()) +"\n \n"
    f.write(init_msg)
    f.close()

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