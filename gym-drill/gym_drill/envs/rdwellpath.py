import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Screen size constants
SCREEN_X = 1000
SCREEN_Y = 1000
SCREEN_Z = 1000

STEP_SIZE = 10.0

MAX_ANG_VEL = 3.0 * (np.pi / 180)
MAX_ANG_ACC = 1.0 * (np.pi / 180)

ANG_ACC_INCREMENT = 0.25 * (np.pi / 180)

PLOT_ENABLED = False

MIN_TAR_RAD = 40
MAX_TAR_RAD = 70

SHOW_GENERATED_PLOTS = False

class TargetSet():
    def __init__(self, tlist, hlist):
        self.tlist = tlist
        self.hlist = hlist

def targetset_to_file(file_name, tset_list):
    with open(file_name, "w") as file:

        # For each set
        for tset in tset_list:
            # Add all targets as strings to list
            target_str_list = []
            for target in tset.tlist:
                # Print values from target
                target_str = [str(int(i)) for i in target]
                separator = ","
                string = separator.join(target_str)
                target_str_list.append(string)
            
            # Add all targets with semicolons between them
            separator = ";"
            tmp = separator.join(target_str_list)
            file.write(tmp)
            # Print dash
            file.write("-")

            # Print hazards
            hazard_str_list = []
            for hazard in tset.hlist:
                # Print values from hazard
                hazard_str = [str(int(i)) for i in hazard]
                separator = ","
                string = separator.join(hazard_str)
                hazard_str_list.append(string)
            
            separator = ";"
            tmp = separator.join(hazard_str_list)
            file.write(tmp)
            file.write("\n")

def get_random_radius():
    return np.random.uniform(MIN_TAR_RAD, MAX_TAR_RAD)

def plot_sphere(ax, pos=[0, 0, 0], rad=1, name='', color="blue"):
    # Make data
    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 8)
    x = pos[0] + rad * np.outer(np.cos(u), np.sin(v))
    y = pos[1] + rad * np.outer(np.sin(u), np.sin(v))
    z = pos[2] + rad * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_wireframe(x, y, z, color=color)
    ax.text(pos[0] + rad, pos[1] + rad, pos[2] + rad, name, None)

# Return angle from normal distribution, that may not deviate
# from mean more than dictated by limit.
def generate_random_angle(mean, standard_deviation, limit):
    angle = (mean - limit - 1)
    while not (mean - limit) < angle < (mean + limit):
        angle = np.random.normal(mean, standard_deviation)
    return angle


def generate_target_list(n_targets, start_pos, min_len, max_len):

    path_created = False
    while not path_created:
        # Initialise to start values
        pos = np.asarray(start_pos)
        pos_list = [pos]

        azi = generate_random_angle(0.25*np.pi, 0.5, 1.2)
        azi_ang_vel = 0
        azi_ang_acc = 0

        inc = generate_random_angle(0.1*np.pi, 0.5, 1.2)
        inc_ang_vel = 0
        inc_ang_acc = 0

        xs = []
        ys = []
        zs = []

        # Calculate each step
        for i in range(max_len):
            # Randomly choose steering action
            action = np.random.randint(0,4)

            if action == 0 and inc_ang_acc < MAX_ANG_ACC:
                inc_ang_acc += ANG_ACC_INCREMENT
            elif action == 1 and inc_ang_acc > -MAX_ANG_ACC:
                inc_ang_acc -= ANG_ACC_INCREMENT
            elif action == 2 and azi_ang_acc < MAX_ANG_ACC:
                azi_ang_acc += ANG_ACC_INCREMENT
            elif action == 3 and azi_ang_acc > -MAX_ANG_ACC:
                azi_ang_acc -= ANG_ACC_INCREMENT

            # Then calculate resulting positional values
            if abs(inc_ang_vel + inc_ang_acc) < MAX_ANG_VEL:
                inc_ang_vel += inc_ang_acc
            
            if abs(azi_ang_vel + azi_ang_acc) < MAX_ANG_VEL:
                azi_ang_vel += azi_ang_acc

            inc += inc_ang_vel
            azi += azi_ang_vel

            # Create 3D step vector with length STEP_SIZE
            step = STEP_SIZE * np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

            # Add step to position vector
            pos = pos + step

            # Add position to pos_list
            pos_list.append(pos)
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

            # If position is outside bounds
            if not (0 < pos[0] < SCREEN_X and 0 < pos[1] < SCREEN_Y and 0 < pos[2] < SCREEN_Z):
                # If i > min_len
                if i > min_len:
                    path_created = True
                    break
                else:
                    # Start over
                    break
            if i + 1 == max_len:
                path_created = True
                break # Fail safe   

    
    # Pick from only every x points to avoid target balls being too close

    every = 10
    n_steps_skipped = int(0.1 * len(pos_list))
    #n_steps_skipped = 1
    #print("len pos list", len(pos_list))

    idx_arr = np.random.choice(a=range(n_steps_skipped, len(pos_list), every), size=n_targets, replace=False)

    idx_arr = np.sort(idx_arr)

    #print(idx_arr)


    ret_arr = np.zeros((n_targets, 4))
    for i in range(n_targets):
        ret_arr[i][0] = pos_list[i][0]
        ret_arr[i][1] = pos_list[i][1]
        ret_arr[i][2] = pos_list[i][2]

        ret_arr[i][3] = get_random_radius()

    if SHOW_GENERATED_PLOTS:
        # Plot well path
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(0, SCREEN_X)
        ax.set_ylim(SCREEN_Y, 0)
        ax.set_zlim(SCREEN_Z, 0)

        target_num = 1
        for idx in idx_arr:
            plot_sphere(ax, pos_list[idx], rad=40, name=str(target_num), color="#32cd32")
            target_num += 1
        
        # Get well path
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)

        ax.plot(xs, ys, zs)
        plt.show()

    return ret_arr


def random_targetset_to_file(file_name, n_sets, n_targets, start_pos, min_len, max_len):
    tset_list = []
    for i in range(n_sets):
        # Generate list of targets
        target_list = generate_target_list(n_targets, start_pos, min_len, max_len)
        # Add list of targets to instance of TargetSet class (no hazards added)
        tset = TargetSet(target_list, [])
        # Add instance of targetset class to target set list
        tset_list.append(tset)
        

    # Write it to file
    targetset_to_file(file_name, tset_list)


if __name__ == "__main__":
    random_targetset_to_file("deletethis.txt", 5, 5, [200, 100, 100], 120, 200)


