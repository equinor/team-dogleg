## Custom Well Trajectory Environment

Gym-Drill is a custom, OpenAI Gym compatible environment modelling a subsurface reservoar. Its main purpose is to be used in a reinforcement learning context to train an agent to find the best path from a given starting point, to a set of target balls. The environment inflicts a curvature constraint (dogleg severity limitation) on the agents well path. An option to include Hazards (hard constraints that the path cannot intersect) into the subsurface environment also exist.

### Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Advanced use](#advanced-use)
    - [Overwriting the starting conditions](#overwriting-the-starting-conditions)
    - [Toggle hazards, Monte Carlo simulated training and episode log](#toggle-hazards,-monte-carlo-simulated-training-and-episode-log)
    - [Adjust environment parameters](#adjust-environment-parameters)
    - [Key functions to utilize when training](#key-functions-and-attributes-to-utilize-when-training)

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
where `STARTLOCATION` is of type [**Coordinate**](gym_drill/envs/Coordinate.py) and `BIT_INITIALIZATION` is a list/tuple on the form ```[initial_azimuth_angle, initial_azimuth_angular_velocity, initial_azimuth_angular_acceleration,initial_inclination_angle, initial_inclination_angular_velocity, initial_inclination_angular_acceleration```. An example of creating an environment with custom parameters would be:

```python
import gym
import gym_drill
import numpy as np 

from gym_drill.envs.customAdditions import Coordinate

STARTLOCATION = Coordinate(0.0,0.0,0.0)
BIT_INITIALIZATION = [0.0,0.0,0.0,0.0,0.0,0.0]

env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
```
#### Toggle hazards, Monte Carlo simulated training and episode log

In addition to [overwriting the starting conditions] of the agent (bit), there exist options to toggle hazards in the environment, to train with Monte Carlo simulated environments in order ensure the existence of a feasible path and a episode based log that gives you realtime updates regarding the training.

- Toggle hazards by passing ``activate_hazards = True`` as a keyword argument to the ``make()`` function. This will enrich the environment with hazards of amount and size as specified in the [environment config file](gym_drill/envs/environment_config.py). See the [Adjust environment parameters](#adjust-environment-parameters) section for details. By default this is set to ``True``

- Toggle Monte Carlo simulated training by passing ``monte_carlo = True`` as a keyword argument to the ``make()`` function. This will ensure that an agent training in the environment always will be exposed to an environment where a feasible path to all targets exist. This is done by first generating a set of random paths and then populate those paths with targets. The details of the Monte Carlo simulation is specified in the [environment config file](gym_drill/envs/environment_config.py). See the [Adjust environment parameters](#adjust-environment-parameters) section for details. By default this is set to ``False``

- Toggle loading of Monte Carlo generated environment by passing ``activate_hazards = True`` as a keyword arugment to the ``make()`` function. If ``True`` togheter with ``monte_carlo`` then it will not generate a new set of Monte Carlo simulated environments, put load form an existing set. It is recommended to use when plotting trained agent to avoid having to generate a new set of Monte Carlo environments

- Toggle the episode log by passing ``activate_log == True`` as a keyword argument to the ``make()`` function. This will write the amount of steps and total reward from each episode to a file named "drill_log.txt". This log will contain the total amount of steps **NOTE: Using the log will greatly reduce performance during training.** It is recommended that the log is used when tweaking the reward system or during very thorough training. By default this is set to ``False``.  

As an example, if you want to turn of hazards and Monte Carlo simulated training, but see behind the scenes magic written in the log, you would do

```python
env = gym.make("drill-v0",activate_hazards = False,monte_carlo = False,activate_log = True)
```

#### Adjust environment parameters

The environment and an agent exeperience in the environment is described by a set of variables that control physical attributes, movement limitations, rewards, what is contained in the observation space and more. These are all stored in the [environment_config](gym_drill/envs/environment_config.py) file. If you feel like changing aspects of the environment for yourself by tweaking these variables all you have to do is update the values inside this file.
#### Key functions and attributes to utilize when training
As the environment is OpenAI gym compatible it has all the attributes and functions you would expect to be in an OpenAI gym environement pr [the documentation](https://gym.openai.com/docs/). These include, but are not limited to:

- A ``reset()`` function which resets the environment to its initial state. Note that even if you are [overwriting the starting conditions](#overwriting-the-starting-conditions) the Azimuth and Inclination angle will be drawn randomly, to ensure that the training is not beeing overfitted for one particular starting angle.
- A ``step()`` function that accepts an ``action`` and executes that action in the environment. In accordance with the documentation the ``step()`` function returns:
    - An ``observation`` object, containing the new state of the environment after the action has been executed
    - A ``reward`` (float), which is the reward the agent recieved for exectuing the particular action 
    - A ``done`` signal (boolean) indicating wheter or not the episode is over
    - An ``info`` (dictionary) message containing diagnostic information useful for debugging. 

The only deviation from the functions describes in the documentation is that the ``render()`` function that most OpenAI gym environment use to display the environment has been replaced with two seperate functions. ``display_planes()`` for displaying the horizontal (xy) and vertical (zy) planes and ``display_3d_environment()`` which displays the path and environment in a 3D plot.

*Last updated 28.07.2020*