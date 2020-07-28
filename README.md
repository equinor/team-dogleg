![alt text](https://dwglogo.com/wp-content/uploads/2019/03/2500px-equinor_logo.png)
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
- [Using the custom environment](#using-the-custom-environment)
- [Adjust training parameters](#adjust-training-parameters) 


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
~$ docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app -p 0.0.0.0:6006:6006 -p  0.0.0.0:8988:8988 --name auto_well_path_running auto_well_path
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

More algorithms comming soon!

## Using the custom environment

If it is desirable to perform your own non-stable-baselines training or just customize the training in general, using the custom made OpenAI Gym compatible environment, see the custom gym environment [documentation](gym-drill/README.md).

## Adjust training parameters

To tweak parameters related to the training in stable-baselines, see the [agent_training](gym-drill/agent_training.py) file together with the stable baselines documentation for the [specific algorithm you want to alter](#available-algorithms).

