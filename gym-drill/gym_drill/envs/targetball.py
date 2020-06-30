import numpy as np

INIT_HIT_REW = 100 # Reward for first hit on a target ball
CTD_HIT_REW = 5    # Rew. for each consecutive hit on tball.
APPROACH_REW = DRILL_SPEED * 7 # Rew. for approaching tball.
MOVE_REW = DRILL_SPEED * -1    # Penalty for each move.
STEER_REW = -10    # Penalty for altering ang. velocity.
OOF_REW = -500     # Out of field penalty (leaving the screen).

DRILL_SPEED = 5.0

# When drill has hit the target, how much does it need to keep approaching
# centre of target in order to continue receiving consecutive hit rewards?
MIN_CTD_APPR = 10 

def angle_between_vectors(v1, v2):
    # Return angle in radians between two 2D vectors
    rad_ang = np.math.atan2(np.linalg.det([v1,v2]), np.dot(v1,v2))
    return rad_ang

def dist_between(pos1, pos2):
    # Return distance between pos1 and pos2
    # Both must be numpy vectors
    return np.linalg.norm(pos1 - pos2)


class TargetBall():
    def __init__(self, x, y, rad, rew_weight):
        self.pos = np.array([x, y])
        self.rad = rad    # Radius of ball
        self.hit = False  # Has the drill ever hit the target?
        self.left = False # Has the drill ever left the target after hitting it?
        self.closest_yet = None # How close has the drill ever been to the target?

    def get_reward(self, drill_pos, drill_heading, drill_ang_acc):
        # When the agent approaches its next target, it is rewarded.
        # When the agent hits the ball, it gets a one-time reward.
        # As long as the agent gets closer to the centre of a target it has hit,
        # it continues getting rewarded.
        
        reward = 0

        # If bit is inside target ball
        if dist_between(self.pos, drill_pos) < self.rad:

            # If bit has not hit target before
            if not self.hit:
                # Set hit to True
                self.hit = True
                # Give one-time reward for hitting target
                reward += INIT_HIT_REW

                # Start recording how close we are to the centre of target.
                self.closest_yet = dist_between(self.pos, drill_pos)

            # If bit has hit target
            else:
                # If target has not been left
                if not self.left:

                    # While we are still approaching target centre, give reward.
                    # Note: It must approach by a certain amount in order to count.
                    if dist_between(self.pos, drill_pos) - self.closest_yet > MIN_CTD_APPR:
                        reward += CTD_HIT_REW
                        self.closest_yet = dist_between(self.pos, drill_pos)
                    
                    ## When we aren't approaching target anymore, set self.left = True
                    else:
                        self.left = True

        # If bit is not in target
        else:
            # If bit has not hit target
            if not self.hit:
                # Give approach reward
                reward += self.get_approach_reward(drill_pos, drill_heading)

            # If bit has hit target
                ## Target is no longer relevant
    
        # Give a negative reward for each move.
        reward += MOVE_REW

        # Give a negative reward for steering.
        if drill_ang_acc != 0:
            reward += STEER_REW

        # Return the reward
        return reward

    def get_approach_reward(self, drill_pos, drill_heading):
        # Find the vector that points from drill towards target.
        # Compare angle between this vector and heading vector.
        # This angle decides the reward:
        
        # Reward is multiplied by 0 if angle is pi/2
        # multiplied by 1 if angle is 0
        # multiplied by -1 if angle is pi

        approach_vector = self.pos - drill_pos
        angle = angle_between_vectors(approach_vector, drill_heading)

        reward_factor = np.cos(angle)
        return reward_factor * APPROACH_REW


def angle_between_vectors(v1, v2):
    # Return angle in radians between two 2D vectors
    rad_ang = np.math.atan2(np.linalg.det([v1,v2]), np.dot(v1,v2))
    return rad_ang

def dist_between(pos1, pos2):
    # Return distance between pos1 and pos2
    # Both must be numpy vectors
    return np.linalg.norm(pos1 - pos2)