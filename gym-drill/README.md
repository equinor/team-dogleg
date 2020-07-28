## Custom Well Trajectory Environment

Gym-Drill is a custom, OpenAI Gym compatible environment modelling a subsurface reservoar. Its main purpose is to be used in a reinforcement learning context to train an agent to find the best path from a given starting point, to a set of target balls. The environment inflicts a curvature constraint (dogleg severity limitation) on the agents well path. An option to include Hazards (hard constraints that the path cannot intersect) into the subsurface environment also exist. 

### Installation

Using the custom environment requires [OpenAI Gym](https://gym.openai.com/) (duh), [matplotlib](https://matplotlib.org/) for displaying well paths and [numpy](https://numpy.org/) for number crunching. If not already installed, run from the terminal:

```
~$ pip install gym numpy matplotlib
```

To install the custom gym environment, navigate to the ``gym-drill`` folder

```cmd
~$ cd gym-drill
```

and run:

```cmd
~$ pip install -e .
```

Thats it. Happy training!
### Quick start

To use the environment both ``gym`` and ``gym_drill`` must be imported to Python file where an instance of the environment will be created. Notice the difference between the ``gym-drill`` and the ``gym_drill`` folder (TODO: what is the difference, haha). To create an instance of the environment, use OpenAI Gyms ``make()`` function and pass the environment name ``drill-v0`` as argument

```python
import gym
import gym_drill

env = gym.make("drill-v0")
```
This will create an instance of the environment with the agent (bit) starting in position **(1000,1000,0)** with the values for the  [Inclination and Azimuth angles](https://www.researchgate.net/figure/a-Azimuth-and-inclination-when-drilling-a-directional-well-b-The-azimuth-of-a_fig2_320511730) beeing drawn uniformly with intervals (0,pi/4) and (0,2pi) respectively. See [register function](gym_drill/__init__.py) for details.

### Advanced use

There are several parameters that can be adjusted to change the environment, both the "physical" aspects and the rewards for different actions.
 
#### Overwriting the starting conditions
To initialize an instance of the environment with your own specificed parameters do

```python
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
```
where `STARTLOCATION` is of type [**Coordinate**](gym_drill/envs/Coordinate.py) and `BIT_INITIALIZATION` is a list/tuple on the form `[initialAngle, initialAngularVelocity, initialAngularAcceleration]`. An example of creating an environment with custom parameters would be:

```python
import gym
import gym_drill
import numpy as np 

from gym_drill.envs.customAdditions import Coordinate

STARTLOCATION = Coordinate(0.0,0.0,0.0)
BIT_INITIALIZATION = [0.0,0.0,0.0]

env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
```
#### Toggle hazards, Monte Carlo simulated training and episode log

In addition to [overwriting the starting conditions] of the agent (bit), there exist options to toggle hazards in the environment, to train with Monte Carlo simulated environments in order ensure the existence of a feasible path and a episode based log that gives you realtime updates regarding the training.

- Toggle hazards by passing ``activate_hazards = True`` as a keyword argument to the ``make()`` function. This will enrich the environment with hazards of amount and size as specified in the [environment config file](gym_drill/envs/environment_config.py). See the [Adjust environment parameters](#adjust-environment-parameters) section for details. By default this is set to ``True``

- Toggle Monte Carlo simulated training by passing ``monte_carlo = True`` as a keyword argument to the ``make()`` function. This will ensure that an agent training in the environment always will be exposed to an environment where a feasible path to all targets exist. This is done by first generating a set of random paths and then populate those paths with targets. The details of the Monte Carlo simulation is specified in the [environment config file](gym_drill/envs/environment_config.py). See the [Adjust environment parameters](#adjust-environment-parameters) section for details. By default this is set to ``False``

- Toggle the episode log by passing ``activate_log == True`` as a keyword argument to the ``make()`` function. This will write the amount of steps and total reward from each episode to a file named "drill_log.txt". This log will contain the total amount of steps **NOTE: Using the log will greatly reduce performance during training.** It is recommended that the log is used when tweaking the reward system or during very thorough training. By default this is set to ``False``.  

As an example, if you want to turn of hazards and Monte Carlo simulated training, but see behind the scenes magic written in the log, you would do

```python
env = gym.make("drill-v0",activate_hazards = False,monte_carlo = False,activate_log = True)
```

#### Adjust environment parameters
physical attributes, movement limitations, rewards, what is contained in the observation space 
#### Key functions to utilize when training

*Last updated 28.07.2020*