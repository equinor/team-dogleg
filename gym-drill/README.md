## Custom Well Trajectory Environment

This is a custom environment modelling the trajectory of a well. Compatible with OpenAI Gym

### Usage

In addition to importing `gym` and `gym_drill`, you must import the custom coordinate class. This is included in the environemnt and can be accomplished by the following import statement

```python
import gym
import gym_drill
from gym_drill.envs.customAdditions import Coordinate
```


To create an instance of the environment do

```python
env = gym.make("drill-v0")
env.initParameters(startLocation,targets,bitInitialization[3])
```

where `startLocation` is a Coordinate type representing the starting location of the bit. `targets` is a tuple, where each element is another tuple consisting of a Coordinate and a raduis which completely specifies a target ball `tuple(tuple(Coordinate,raduis))`. `targetRadius`indicates the radius of the target ball and is a number (float or int). Finally, `bitInitialization` is a tuple (*could be list also, but tuple ensures immutability*) containing the initial angle of the bit, the initial angular velocity and angular acceleration of the bit `tuple(angle,angularVelocity,angularAcceleration)`.