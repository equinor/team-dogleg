import numpy as np

# Max values for angular velocity and acceleration
MAX_ANGVEL = 0.1
MAX_ANGACC = 0.05

# The allowed increment. We either add or remove this value to the angular acceleration
ANGACC_INCREMENT = 0.01
DRILL_SPEED = 5.0

# Screen size, environment should be square
SCREEN_X = 2000
SCREEN_Y = 2000

# Target specs
TARGET_BOUND_X = [0.25*SCREEN_X,0.85*SCREEN_X]
TARGET_BOUND_Y = [0.2*SCREEN_Y,0.75*SCREEN_Y]
TARGET_RADII_BOUND = [20,50]

NUM_TARGETS = 4
TARGET_WINDOW_SIZE = 3
NUM_MAX_STEPS = ((SCREEN_X+SCREEN_Y)/DRILL_SPEED)*1.3

# Rewards
FINISHED_EARLY_FACTOR = 1 # Point per unused step

# Hazard specs. Can be in entire screen
HAZARD_BOUND_X = [0,SCREEN_X]
HAZARD_BOUND_Y = [0,SCREEN_Y]
HAZARD_RADII_BOUND = [20,50]

NUM_HAZARDS = 4

# Observation space specs
SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y] # x_low,x_high,y_low,y_high
BIT_BOUNDS = [0,2*np.pi,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGACC,MAX_ANGACC] #
HAZARD_BOUNDS = [HAZARD_BOUND_X,HAZARD_BOUND_Y,HAZARD_RADII_BOUND]
TARGET_BOUNDS = [TARGET_BOUND_X,TARGET_BOUND_Y,TARGET_RADII_BOUND]

# Additional data
DIAGONAL = np.sqrt(SCREEN_X**2 + SCREEN_Y**2)
TARGET_DISTANCE_BOUND = [0,DIAGONAL]
RELATIVE_ANGLE_BOUND = [-np.pi,np.pi]
EXTRA_DATA_BOUNDS = [TARGET_DISTANCE_BOUND,RELATIVE_ANGLE_BOUND] # [Distance, angle between current direction and target direction]

# All reward values go here. The reward will add these values. Make sure signs are correct!
STEP_PENALTY = -0.0
ANGULAR_VELOCITY_PENALTY = -1.0
ANGULAR_ACCELERATION_PENALTY = -2.0
OUTSIDE_SCREEN_PENALTY = -30.0
TARGET_REWARD = 100.0
HAZARD_PENALTY = -100.0
ANGLE_REWARD_FACTOR = 4

NUM_MAX_STEPS = ((SCREEN_X+SCREEN_Y)/DRILL_SPEED)*1.3
FINISHED_EARLY_FACTOR = 1 # Point per unused step