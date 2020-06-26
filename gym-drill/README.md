## Custom Well Trajectory Environment

This is a custom environment modelling the trajectory of a well. Compatible with OpenAI Gym

### Installation

Using the custom environment requires [OpenAI Gym](https://gym.openai.com/) and [Numpy](https://numpy.org/). If not already installed run from the terminal:

```
$~ pip install gym numpy
```

To install the custom gym environment run:

```python
$~ pip install -e .
```

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
```
This will create an environment with the bit starting in position **(100,300)** with and intial angle relative to y-axis of 3pi/4 (measured counterclockwise from the y-axis). The angular acceleration and velocity will both be zero. See [register function for details](gym_drill/__init__.py).

To initialize an instance of the environment with your own specificed parameters do

```python
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
```
where `STARTLOCATION` is of type [**Coordinate**](gym_drill/envs/customAdditions.py) and `BIT_INITIALIZATION` is a list/tuple on the form `[initialAngle, initialAngularVelocity, initialAngularAcceleration]`. And example of creating an environment with custom parameters would be:

```python
import gym
import gym_drill
import numpy as np 

from gym_drill.envs.customAdditions import Coordinate

STARTLOCATION = Coordinate(0.0,0.0)
BIT_INITIALIZATION = [0.0,0.0,0.0]

env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
```

*Last updated 26.06.2020*