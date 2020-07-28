"""
This config file completely describes the "physical" aspects of the environemnt aswell as its rewards system
"""

import numpy as np

# Limits on the angles. They should match the limits of angles describing a sphere in spherical coordinates
MIN_INCL_ANGLE = 0
MAX_INCL_ANGLE = np.pi

MAX_AZIMUTH_ANGLE = 2*np.pi

# Max values for angular velocity and acceleration for both angles
MAX_ANGVEL = 0.1
MAX_ANGACC = 0.02

# The allowed increment. We either add or remove this value to both angles angular acceleration 
ANGACC_INCREMENT = 0.02

# Step size. For each step the bit position gets updated with DRILL_SPEED multiplied with cos/sin of one of the angles
DRILL_SPEED = 10

# Screen size, environment should be square
SCREEN_X = 2000
SCREEN_Y = 2000
SCREEN_Z = 2000

# Step budget agent has available 
NUM_MAX_STEPS = ((SCREEN_X+SCREEN_Y+SCREEN_Z)/DRILL_SPEED)*1.5

# Target specs specifying where a target can exist
TARGET_BOUND_X = [0.25*SCREEN_X,0.75*SCREEN_X]
TARGET_BOUND_Y = [0.25*SCREEN_Y,0.75*SCREEN_Y]
TARGET_BOUND_Z = [0.40*SCREEN_Z,0.85*SCREEN_Z]
TARGET_RADII_BOUND = [40,50]

NUM_TARGETS = 8
TARGET_WINDOW_SIZE = 3

# Hazard specs. Can exist in entire screen
HAZARD_BOUND_X = [0,SCREEN_X]
HAZARD_BOUND_Y = [0,SCREEN_Y]
HAZARD_BOUND_Z = [0,SCREEN_Z]
HAZARD_RADII_BOUND = [100,150]

NUM_HAZARDS = 8
HAZARD_WINDOW_SIZE = 1

# Common specs for both targets and hazards
VER_DIST_BOUND = [-SCREEN_Z, SCREEN_Z] # bounds for the vertical distance
HOR_DIST_BOUND = [0,np.sqrt(SCREEN_X**2+SCREEN_Y**2)] # bounds for the horizontal distance
REL_AZIMUTH_BOUND = [-np.pi,np.pi]

# Observation space specs (vectorized bounds)
SPACE_BOUNDS = [0,SCREEN_X,0,SCREEN_Y,0,SCREEN_Z] 
BIT_BOUNDS = [0,2*np.pi,0,np.pi,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGVEL,MAX_ANGVEL,-MAX_ANGACC,MAX_ANGACC,-MAX_ANGACC,MAX_ANGACC]
HAZARD_BOUNDS = [VER_DIST_BOUND,HOR_DIST_BOUND,REL_AZIMUTH_BOUND,HAZARD_RADII_BOUND]
TARGET_BOUNDS = [VER_DIST_BOUND,HOR_DIST_BOUND,REL_AZIMUTH_BOUND,TARGET_RADII_BOUND]

# Rewards
STEP_PENALTY = -0.0
ANGULAR_VELOCITY_PENALTY = 0.0
ANGULAR_ACCELERATION_PENALTY = 0.0
OUTSIDE_SCREEN_PENALTY = 0.0
TARGET_REWARD = 100.0
HAZARD_PENALTY = -200.0
ANGLE_REWARD_FACTOR = 0.5
INCLINATION_REWARD_FACTOR = 0.5
FINISHED_EARLY_FACTOR = 1 # Point per unused step

# Generating environments with a Monte Carlo simulation
NUM_MONTE_CARLO_ENVS = int(1e4)
MC_PATH_LENGTH_BOUND = [250,350]
ENVIRONMENT_FILENAME = "environments.txt"