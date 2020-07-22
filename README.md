# team-dogleg
Team repo for Team Dogleg - Virtual summer internship 2020

## Run locally
It is recommended to [run using docker](run-with-docker-using-windows), however if you don't have Docker available it is possible to run it *normally* on you computer.

### Installations

Install the requried dependencies:

```
~$ pip install -r requirements.txt
```
Note that some of the packages are only compatable with Python versions 3.7 or lower. Verify your Python verison with ```python -v``` and downgrade or consider [running with Docker](run-with-docker-using-windows) if the version is insufficient.

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
- The ``<algorithm>`` argument specifies which algorithm to use when training. This argument is not case sensitive. For valid algorithms to pass as an argument see [available algorithms](available-algorithms). When retraining and loading this algorithm must match the algorithm of the existing agent. To keep track of what algorithm is used on a particular agent a naming convention indicating this is recommended.  



 In addition to the three required arguments there are additional optional arguments, ``<timesteps>`` and ``<new_save_name>``, when running with the train or retrain argument.

 - The ``<timesteps>``argument can be used with both train and retrain and specifies the total amount of timesteps the training should last. If not specified a total of 10 000 timesteps will be used.

 - The ``<new_save_name>`` argument can be used with retrain and specifies the name of the newly trained agent. If not specified the newly trained will overwrite the old agent named ``<save_name>``

The arguments are summarized in the table below.


| Action                                             	| Argument 	| Required arguments              	| Optional arguments           	|
|----------------------------------------------------	|----------	|---------------------------------	|--------------------------------	|
| Train new agent                                    	| train    	| \<save_name>  \<algorithm_used> 	| \<timesteps>                   	|
| Further train existing agent                       	| retrain  	| \<save_name>  \<algorithm_used> 	| \<timesteps>  \<new_save_name> 	|
| Load and display existing agent in the environment 	| load     	| \<save_name>  \<algorithm_used> 	| -                              	|

### Examples
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
### Installations

Running with Docker is recommended to ensure a satisfying runtime environment. If you haven't already, install [Docker for Windows](https://docs.docker.com/docker-for-windows/install/).

### Building and running the Docker container

For user convenience all interactions with the Docker container is done from a PowerShell script. Should detalis regarding the specific Docker commands be of interest the user is reffered to [the script itself](run.ps1). It is recommended that you run the script from a PowerShell terminal as administrator. To get started, make sure you are allowed to execute scripts by setting the execution policy to unrestriced. In PowerShell, run

```
~$ Set-ExecutionPolicy Unrestricted
```

#### Building the Docker container
Build the Docker container using the ``-build`` flag:

```
~$ ./run.ps1 -build 
```
This will create a Docker container named ```auto_well_path```.

#### Run the Docker container
Having built the container, you can run it be passing the ``-run`` flag. Running the container will in this case be the same as running the program. Therefore the arguments introduced in the [running the program locally section](run-the-program) must be passed.

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

## Available algorithms

The following algorithms to train agents are available 

- **DQN**. For more information see [the stable-baselines documentation](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html). To specify the DQN algorithm when running the program pass "DQN" or "dqn" as the ``<algorithm>`` argument.
- **PPO2**. For more information see [the stable-baselines documentation](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html). To specify the PPO2 algorithm when running the program pass "PPO2" or "ppo2" as the ``<algorithm>`` argument.

More algorithms comming soon!

## Life advice when using docker

- List all your docker images by running `docker image`
- Delete a docker image using `docker image rm [IMAGE ID]`. You can find the ID of the image by listing all images using `docker image`
