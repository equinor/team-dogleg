# This file is responsible for generating sets of targetballs that are
# theoretically possible for the agent to reach. This is done through
# simulation of a pseudo-agent taking random actions, generating a path,
# and then placing targetballs at random locations on the path.

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym_drill.envs import environment_config as cfg

MC_FIT_TRIES = 10 # How many tries to give for each set of hazards before trying with new ones.
CUT_OFF_START = True # Should we discard the first X% of the steps?
CUT_OFF_PERCENT = 10 # Only relevant if CUT_OFF_START == True
SHOW_GENERATED_PLOTS = False # Show generated plots. Should be False.


# Return angle from normal distribution, that may not deviate
# from mean more than dictated by limit.
def get_random_angle(mean, standard_deviation, limit):
    angle = (mean - limit - 1)
    while not (mean - limit) < angle < (mean + limit):
        angle = np.random.normal(mean, standard_deviation)
    return angle

def get_hazards(n_hazards):

    hazards = np.zeros((3, n_hazards))

    # Generate n x-coords, y-coords and z-coords
    hazards[0] = np.random.randint(cfg.HAZARD_BOUND_X[0], cfg.HAZARD_BOUND_X[1], n_hazards)
    hazards[1] = np.random.randint(cfg.HAZARD_BOUND_Y[0], cfg.HAZARD_BOUND_Y[1], n_hazards)
    hazards[2] = np.random.randint(cfg.HAZARD_BOUND_Z[0], cfg.HAZARD_BOUND_Z[1], n_hazards)
    hazards = hazards.T

    # Generate radiuses
    radiuses = np.random.randint(cfg.HAZARD_RADII_BOUND[0], cfg.HAZARD_RADII_BOUND[1], n_hazards)

    return hazards, radiuses

def plot_sphere(ax, array, name='', color="blue"):
    # Make data
    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 8)
    x = array[0] + array[3] * np.outer(np.cos(u), np.sin(v))
    y = array[1] + array[3] * np.outer(np.sin(u), np.sin(v))
    z = array[2] + array[3] * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_wireframe(x, y, z, color=color)
    # Show name
    ax.text(array[0] + array[3], array[1] + array[3], array[2] - array[3], name, None)


# This function will return two arrays.
# The first 4D-array contains 3D coordinates + radiuses for targets
# The second 4D-array contains 3D coordinates + radiuses for hazards
def generate_targets_hazards(n_targets, n_hazards, min_coords, max_coords, min_pathlen, max_pathlen, start_pos):

    min_coords = np.asarray(min_coords)
    max_coords = np.asarray(max_coords)
    start_pos = np.asarray(start_pos)

    path_created = False

    while not path_created:

        # Create a set of hazards, if there should be any
        if n_hazards > 0:
        
            haz_coords, haz_radiuses = get_hazards(n_hazards)
            # Create a KD-tree from the hazard coordinates
            haz_coord_tree = spatial.cKDTree(haz_coords)

        # We then give the algorithm MC_FIT_TRIES tries to try to fit a path around these hazards.
        for _ in range(MC_FIT_TRIES):

            n_steps = 0

            # Initialise positional variables
            pos = np.asarray(start_pos)

            # Azimuth values
            azi = get_random_angle(0.25*np.pi, 0.5, 1.2)
            azi_ang_vel = 0
            azi_ang_acc = 0

            # Inclination values
            inc = get_random_angle(0.1*np.pi, 0.5, 1.2)
            inc_ang_vel = 0
            inc_ang_acc = 0

            # Initialise position array
            pos_arr = np.zeros((max_pathlen, 3))
            pos_arr[0] = pos

            # Initialise random action array
            action_arr = np.random.randint(0, 5, (max_pathlen))

            # Start calculating steps
            for j in range(max_pathlen):
                # Update angular acceleration values as dictated by random action
                if action_arr[j] == 0 and inc_ang_acc < cfg.MAX_ANGACC:
                    inc_ang_acc += cfg.ANGACC_INCREMENT
                elif action_arr[j] == 1 and inc_ang_acc > -cfg.MAX_ANGACC:
                    inc_ang_acc -= cfg.ANGACC_INCREMENT
                elif action_arr[j] == 2 and azi_ang_acc < cfg.MAX_ANGACC:
                    azi_ang_acc += cfg.ANGACC_INCREMENT
                elif action_arr[j] == 3 and azi_ang_acc > -cfg.MAX_ANGACC:
                    azi_ang_acc -= cfg.ANGACC_INCREMENT
                
                if abs(inc_ang_vel + inc_ang_acc) < cfg.MAX_ANGVEL:
                    inc_ang_vel += inc_ang_acc
                if abs(azi_ang_vel + azi_ang_acc) < cfg.MAX_ANGVEL:
                    azi_ang_vel += azi_ang_acc
                
                inc += inc_ang_vel
                azi += azi_ang_vel

                # Create 3D step vector of length MC_STEP_SIZE
                step_vec = cfg.MC_STEP_SIZE * np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])
                
                # Add step vector to position vector
                pos = pos + step_vec

                # Check if the new position is inside bounds
                if not np.logical_and((min_coords < pos).all(), (pos < max_coords).all()):
                    # If ball left screen but min_pathlen is achieved, we have a valid path
                    if j > min_pathlen:
                        path_created = True
                        break
                    # Else, start over
                    else:
                        break
                
                # If there are hazards, check if they have been hit
                if n_hazards > 0:
                    dist, idx = haz_coord_tree.query(pos)
                    # If the closes hazard has been hit
                    if dist < haz_radiuses[idx]:
                        # if min_pathlen is achieved, we have a valid path
                        if j > min_pathlen:
                            path_created = True
                            break
                        # Else, start over
                        else:
                            break

                # If we reached this point, the path must still be valid.
                # Therefore, we add it to position vector
                pos_arr[j] = pos
                n_steps += 1

                # If max_pathlen is hit, path is successfully created
                if j + 1 == max_pathlen:
                    path_created = True
                    break # Fail safe
            
            if path_created:
                break
    
    if CUT_OFF_START:
        n_steps_skipped = int(n_steps * (CUT_OFF_PERCENT / 100))
    else:
        n_steps_skipped = 0

    # Eliminate all but every x points from being chosen.
    # This helps space the target balls out.
    every = int((cfg.TARGET_RADII_BOUND[0] + cfg.TARGET_RADII_BOUND[1]) / cfg.MC_STEP_SIZE)
    
    # Pick random positions in the pos_arr to use as positions for target balls.
    # This is done by picking random indexes from the array.
    target_idx_arr = np.random.choice(a=range(n_steps_skipped, n_steps, every), size=n_targets, replace=False)
    target_idx_arr = np.sort(target_idx_arr)

    # Pick some random radiuses for the target balls.
    target_radius_arr = np.random.randint(cfg.TARGET_RADII_BOUND[0], cfg.TARGET_RADII_BOUND[1], size=n_targets)


    target_ret_arr = np.zeros((4, n_targets))
    target_ret_arr[0:3] = pos_arr[target_idx_arr].T
    target_ret_arr[3] = target_radius_arr
    target_ret_arr = target_ret_arr.T

    if n_hazards > 0:
        hazard_ret_arr = np.zeros((4, n_hazards))
        hazard_ret_arr[0:3] = haz_coords.T
        hazard_ret_arr[3] = haz_radiuses
        hazard_ret_arr = hazard_ret_arr.T
    else:
        hazard_ret_arr = np.array([])


    # Show generated set for debugging purposes
    if SHOW_GENERATED_PLOTS:
        # Plot well path
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(0, cfg.SCREEN_X)
        ax.set_ylim(cfg.SCREEN_Y, 0)
        ax.set_zlim(cfg.SCREEN_Z, 0)

        
        for i in range(n_targets):
            plot_sphere(ax, target_ret_arr[i], str(i+1), color="#32cd32")
                
        for i in range(n_hazards):
            plot_sphere(ax, hazard_ret_arr[i], color="red")
        

        #ax.plot()
        pos_history_arr = pos_arr[0:n_steps].T
        ax.plot(pos_history_arr[0], pos_history_arr[1], pos_history_arr[2])
        
        plt.show()


    return target_ret_arr, hazard_ret_arr


def targetset_to_file(file_name, target_arr_list, hazard_arr_list):
    with open(file_name, "w") as file:
        for i in range(len(target_arr_list)):

            target_str_list = []
            for j in range(np.shape(target_arr_list)[1]):
                target_str = [str(int(k)) for k in target_arr_list[i][j]]
                separator = ","
                string = separator.join(target_str)
                target_str_list.append(string)
            
            separator = ";"
            tmp = separator.join(target_str_list)
            file.write(tmp)
            file.write("/")

            hazard_str_list = []
            for j in range(np.shape(hazard_arr_list)[1]):
                hazard_str = [str(int(k)) for k in hazard_arr_list[i][j]]
                separator = ","
                string = separator.join(hazard_str)
                hazard_str_list.append(string)
            
            separator = ";"
            tmp = separator.join(hazard_str_list)
            file.write(tmp)
            file.write("\n")


def generate_targets_hazards_to_file(n_targets, n_hazards, min_coords, max_coords, min_pathlen, max_pathlen, start_pos, n_sets, file_name):
    target_arr_list = []
    hazard_arr_list = []

    for _ in range(n_sets):
        target_set, hazard_set = generate_targets_hazards(n_targets, n_hazards, min_coords, max_coords, min_pathlen, max_pathlen, start_pos)
        target_arr_list.append(target_set)
        hazard_arr_list.append(hazard_set)
    
    targetset_to_file(file_name, target_arr_list, hazard_arr_list)

if __name__ == "__main__":
    n_targets = 6
    n_hazards = 6
    min_coords = [0, 0, 0]
    max_coords = [cfg.SCREEN_X, cfg.SCREEN_Y, cfg.SCREEN_Z]
    min_pathlen = 100
    max_pathlen = 340
    start_pos = 140
    n_sets = 5
    file_name = "delete.txt"
    SHOW_GENERATED_PLOTS = True
    
    #generate_targets_hazards_to_file(n_targets, n_hazards, min_coords, max_coords, min_pathlen, max_pathlen, start_pos, n_sets, file_name)
    generate_targets_hazards_to_file(cfg.NUM_TARGETS, cfg.NUM_HAZARDS,
    [cfg.TARGET_BOUND_X[0],cfg.TARGET_BOUND_Y[0],cfg.TARGET_BOUND_Z[0]],
    [cfg.TARGET_BOUND_X[1],cfg.TARGET_BOUND_Y[1],cfg.TARGET_BOUND_Z[1]],
    min_pathlen, max_pathlen,
    [cfg.TARGET_BOUND_X[0],cfg.TARGET_BOUND_Y[0],cfg.TARGET_BOUND_Z[0]],
    n_sets, file_name)
    
    SHOW_GENERATED_PLOTS = False

