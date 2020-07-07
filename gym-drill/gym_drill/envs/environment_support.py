"""
A place to implement smaller custom support functions to be used in the environment
"""
import numpy as np

from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Target import TargetBall


def isWithinTraget(bitPosition,targetPosition,targetRadius):
    return (bitPosition.x - targetPosition.x)**2 + (bitPosition.y - targetPosition.y)**2 < targetRadius

def all_visited(list):
    for i in range(len(list)):
        if list[i]==False:
            return False
    return True

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

# Returns an ordered list of randomly generated targets within the bounds given. 
def _init_targets(num_targets,x_bound,y_bound,r_bound,start_location):
    all_targets = []

    for t in range(num_targets):
        target = _create_unique_random_target(x_bound,y_bound,r_bound,all_targets)
        all_targets.append(target)        
    
    all_targets = _orderTargets(start_location,all_targets)

    return all_targets

# Returns True if t1 or t2 overlap 
def _is_overlapping(t1,t2):
    total_radii = t1.radius + t2.radius
    distance = Coordinate.getEuclideanDistance(t1.center,t2.center)
    return  distance < total_radii

# Creates a uniqe target that does not overlap with any targets in existing_targets
def _create_unique_random_target(x_bound,y_bound,r_bound,existing_targets):
    target_center = Coordinate(np.random.uniform(x_bound[0],x_bound[1]),(np.random.uniform(y_bound[0],y_bound[1] )))
    target_radius = np.random.uniform(r_bound[0],r_bound[1])
    target_candidate = TargetBall(target_center.x,target_center.y,target_radius)

    for target in existing_targets:
        if _is_overlapping(target,target_candidate):
            target_candidate =_create_unique_random_target(x_bound,y_bound,r_bound,existing_targets)
            break

    return target_candidate





