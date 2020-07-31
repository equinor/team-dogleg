# Automate well trajectory planning using artificial intelligence - Team Dogleg
Team repo for Team Dogleg, working on the case *Test reinforcement learning to automate well trajectories* - Equinor Virtual Summer Internship 2020. 

## Table of Contents

- [What it is](#what-it-is)
    - [The problem](#the-problem)
    - [The soltuion](#the-solution)
- [Quickstart](#quickstart)
    - [Run locally](#run-locally)
        - [Installations](#installations-(local))
        - [Run the program](#run-the-program)
        - [Examples of running the program](#examples-of-running-the-program)
    - [Run with Docker using Windows](#run-with-docker-using-windows)
        - [Installations](#installations-(docker))
        - [Building and running the Docker container](#building-and-running-the-docker-container)
    - [Run with Docker using MacOS/Linux](#run-with-docker-using-macos/linux)
- [Available algorithms](#available-algorithms)
- [Adjust training parameters](#adjust-training-parameters) 
- [Using the custom environment](#using-the-custom-environment)
    - [Installation](#installation-(gym-environment))
    - [Quick start](#quick-start)
    - [Advanced use](#advanced-use)
        - [Overwriting the starting conditions](#overwriting-the-starting-conditions)
        - [Toggle hazards, Monte Carlo simulated training and episode log](#toggle-hazards,-monte-carlo-simulated-training-and-episode-log)
        - [Adjust environment parameters](#adjust-environment-parameters)
        - [Key functions to utilize when training](#key-functions-and-attributes-to-utilize-when-training)

- [2D Version](#2d-version)


## What it is

### The problem
Planning the trajectory of a subsurface well is to a large extent a manual process. This is resoruce demanding and the resulting well path is not usually not the optimal path to the desired target location. It is therefore desirable to automate this process with the goal of simplify the well planning process, saving resources and creating more optimal well paths.

### The solution

This repository proposes software attempting to automate the planning of a well path trajectory using reinforcement learning. An agent is trained in an environment modelling a subsurface environment with hazards (no-go areas) and a set of target balls. It uses a [custom made](gym-drill/README.md) environment compatible with [OpenAI Gym](https://gym.openai.com/). The reinforcement learning is implemented using a set of implementations of reinforcement learning algorithms from [stable-baselines](https://stable-baselines.readthedocs.io/en/master/).


## Quickstart

This repository offers two ways to quickly train, retrain or load a trained model using a variety of the [available stable-baslines reinforcement learning algortihms](#available-algorithms). The solution has been dockerized for easy "shipping around-ability", however it is also possible to simply execute the python code outside of Docker on your host machine.  

### Run locally
It is recommended to [run using docker](#run-with-docker-using-windows), however if you don't have Docker available it is possible to run it *normally* on your computer.

### Installations (local)

Install the requried dependencies:

```
~$ pip install -r requirements.txt
```
Note that some of the packages are only compatable with Python versions 3.7 or lower. Verify your Python verison with ```python -v``` and downgrade or consider [running with Docker](#run-with-docker-using-windows) if the version is insufficient.

### Run the program

To run the program, navigate to the ```gym-drill``` folder

```
~$ cd gym-drill
```
and run the program using:

```
~$ python main.py <action> <save_name> <algorithm> <timesteps>* <new_save_name>*
```
An asterisk next to the argument indicates that the argument is not required. There are three required arguments ```<action>```,```<save_name>``` and ```<algorithm>```.

- The ``<action>`` argument has 3 valid values: "train", "retrain" and "load". These will train a new agent, retrain an existing agent or load and display an existing agent, respectively.
- The ``<save_name>`` argument can take any string that is a valid filename on your system. When running with ``<action> == <train>`` this will be the saved name of the newly trained agent. When running with "retrain" or "load" this will be the name of the existing agent to retrain or load.
- The ``<algorithm>`` argument specifies which algorithm to use when training. This argument is not case sensitive. For valid algorithms to pass as an argument see [available algorithms](#available-algorithms). When retraining and loading this algorithm must match the algorithm of the existing agent. To keep track of what algorithm is used on a particular agent a naming convention indicating this is recommended.  



 In addition to the three required arguments there are additional optional arguments, ``<timesteps>`` and ``<new_save_name>``, when running with the train or retrain argument.

 - The ``<timesteps>``argument can be used with both train and retrain and specifies the total amount of timesteps the training should last. If not specified a total of 10 000 timesteps will be used.

 - The ``<new_save_name>`` argument can be used with retrain and specifies the name of the newly trained agent. If not specified the newly trained will overwrite the old agent named ``<save_name>``

The arguments are summarized in the table below.


| Action                                             	| Argument 	| Required arguments              	| Optional arguments           	|
|----------------------------------------------------	|----------	|---------------------------------	|--------------------------------	|
| Train new agent                                    	| train    	| \<save_name>  \<algorithm_used> 	| \<timesteps>                   	|
| Further train existing agent                       	| retrain  	| \<save_name>  \<algorithm_used> 	| \<timesteps>  \<new_save_name> 	|
| Load and display existing agent in the environment 	| load     	| \<save_name>  \<algorithm_used> 	| -                              	|

### Examples of running the program
To further clarify how to properly run the program three examples demonstrating the three possible actions are shown.

1. To **train** a new agent for a total of **600** timesteps using the **DQN** algorithm and saving the agent as **my_trained_agent** do:

```
~$ python main.py train my_trained_agent DQN 600
```
2. To retrain a previously trained agent named **my_previously_trained_agent** for a total of **600** timesteps using the **DQN** algorithm and saving the retrained agent as **my_new_retrained_agent** do:

```
~$ python main.py retrain my_previously_trained_agent DQN 600 my_new_retrained_agent
```
3. To load and view an existing agent named **my_existing_agent** which has been trained with the **DQN** algorithm do:
```
~$ python main.py load my_previously_trained_agent DQN
```

## Run with Docker using Windows
Running with Docker ensures that the code can execute on any computer with Docker installed. For user convenience all interactions with the Docker container is done from a PowerShell script. Should detalis regarding the specific Docker commands be of interest the user is reffered to [the script itself](run.ps1). It is recommended that you run the script from a PowerShell terminal as administrator. To get started, make sure you are allowed to execute scripts by setting the execution policy to unrestriced. In PowerShell, run

```
~$ Set-ExecutionPolicy Unrestricted
``` 
### Installations (Docker)

Running with Docker is recommended to ensure a satisfying runtime environment. If you haven't already, install [Docker for Windows](https://docs.docker.com/docker-for-windows/install/).

### Building and running the Docker container

Building and running the docker container is done by passing different flags as arguments to the ``./run.ps1`` script.

#### Building the Docker container
Build the Docker container using the ``-build`` flag:

```
~$ ./run.ps1 -build 
```
This will create a Docker container named ```auto_well_path```.

#### Run the Docker container
Having built the container, you can run it be passing the ``-run`` flag. Running the container will in this case be the same as running the program. Therefore the arguments introduced in the [running the program locally section](#run-the-program) must be passed.

```
~$ ./run.ps1 -run <action> <save_name> <algorithm> <timesteps>* <new_save_name>*
```

Tensorboard graphs will be running on http://localhost:6006 and if you choose to load trained models, the results will be displayed on http://localhost:5000.

Compared to running the program locally ``python main.py`` has been replaced with ``./run.ps1 -run``.

#### Build and run simultaneously

If you are developing and making changes to the source code, you often have to rebuild the Docker container to see the changes in effect. Frequently re-building and re-running the container as two operations to see the results of small changes can be tiresome. You can avoid this, and build and run in one operation by passing the ``-auto`` flag when running the script. In that case the arguments needed when passing the ``-run`` flag is needed.

```
~$ ./run.ps1 -auto <action> <save_name> <algorithm> <timesteps>* <new_save_name>*
```
#### Abbreviatons
All previously mentioned flags that can be passed to ```run.ps1``` have a corresponding abbreviaton. A complete summary of the different flag options is shown in the table below.

| Action                  | Flag   | Abbreviaton |
|-------------------------|--------|-------------|
| Build container         | -build | -b          |
| Run container           | -run   | -r          |
| Build and run container | -auto  | -a          |


## Run with Docker using MacOS/Linux

Currently we don't have a fancy script for Linux users. To run with linux you would have to manually pass the commands from the ```run.ps1``` script into the terminal.

Build the container using
```cmd
~$ docker build -t auto_well_path .
```

Then, run the conatiner using
```cmd
~$ docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app -p 0.0.0.0:6006:6006 -p  5000:5000 --name auto_well_path_running auto_well_path
```
If wanted, start the tensorboard server displaying the tensorboard graphs in the browser using

```cmd
~$ docker exec -dit auto_well_path_running tensorboard --logdir /usr/src/app/tensorboard_logs --host 0.0.0.0 --port 6006
```

Finally, start the python program using
```
~$ docker exec -it auto_well_path_running python main.py <action> <save_name> <algorithm> <timesteps>* <new_save_name>*
```

where the arguments are the same as the ones in the [running the program locally section](#run-the-program).
## Available algorithms

The following algorithms to train agents are available 

- **DQN**. For more information see [the stable-baselines documentation](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html). To specify the DQN algorithm when running the program pass "DQN" or "dqn" as the ``<algorithm>`` argument.
- **PPO2**. For more information see [the stable-baselines documentation](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html). To specify the PPO2 algorithm when running the program pass "PPO2" or "ppo2" as the ``<algorithm>`` argument. 

    **NB:** *Since the results with the PPO2 algorithm during development have been weak, the support for PPO2 is limited. Functionality for dispaying trained PPO2 agents have not been developed.*

More algorithms comming soon!

## Adjust training parameters

To tweak parameters related to the training in stable-baselines, see the [agent_training](gym-drill/agent_training.py) file together with the stable baselines documentation for the [specific algorithm you want to alter](#available-algorithms).

## Using the custom environment

If it is desirable to perform your own non-stable-baselines training or just customize the training in general, using the custom made OpenAI Gym compatible environment you can use the custom Gym-Drill environment.
<!--## Custom Well Trajectory Environment-->

Gym-Drill is a custom, OpenAI Gym compatible environment modelling a subsurface reservoar. Its main purpose is to be used in a reinforcement learning context to train an agent to find the best path from a given starting point, to a set of target balls. The environment inflicts a curvature constraint (dogleg severity limitation) on the agents well path. An option to include Hazards (hard constraints that the path cannot intersect) into the subsurface environment also exist.

<!--### Table of Contents

- [Installation](#installation-(gym-environment))
- [Quick start](#quick-start)
- [Advanced use](#advanced-use)
    - [Overwriting the starting conditions](#overwriting-the-starting-conditions)
    - [Toggle hazards, Monte Carlo simulated training and episode log](#toggle-hazards,-monte-carlo-simulated-training-and-episode-log)
    - [Adjust environment parameters](#adjust-environment-parameters)
    - [Key functions to utilize when training](#key-functions-and-attributes-to-utilize-when-training)
-->
### Installation (Gym environment)

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

To use the environment both ``gym`` and ``gym_drill`` must be imported to the python file where an instance of the environment will be created. Notice the difference between the ``gym-drill`` and the ``gym_drill`` folder. To create an instance of the environment, use OpenAI Gyms ``make()`` function and pass the environment name ``drill-v0`` as argument

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

from gym_drill.envs.Coordinate import Coordinate

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

## 2D Version

A 2D version of the environment was us as the stepping stone to develop the complete 3D environment. It is contained in the [2D-version folder](2D-version/). Be aware that it is outdated and does contain some bugs

*Last updated 31.07.2020*
