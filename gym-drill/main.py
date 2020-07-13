import gym
import gym_drill
import random
import numpy as np 
import matplotlib.pyplot as plt

from gym_drill.envs.Coordinate import Coordinate

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy as DQN_MlpPolicy
from stable_baselines.deepq.policies import LnMlpPolicy 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, PPO2, A2C, ACER, ACKTR


# Ignore the crazy amount of warnings
import warnings
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Setting up the environment
STARTLOCATION = Coordinate(100,900.0)
BIT_INITIALIZATION = [3.8*np.pi/4,0.0,0.0]

env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
model_name = "deepq_gym-drill-two_random_targets_v0.0"

print("Obs space", env.observation_space)
print("action space", env.action_space)

#Using Stable-Baselines to teach an agent 

#DQN-approach

model_to_load = "DQN_drill_model_2M"
save_as = "DQN_drill_model_4M"
tensorboard_folder = "./algorithm_performance_comparison_v0.3/"
tensorboard_run_name = "DQN_run4_try2"
#Chose one of the two lines below (#1 or #2):
model = DQN(LnMlpPolicy, env, verbose=1, tensorboard_log=tensorboard_folder)           #1) Make a new model
#model = DQN.load(model_to_load, env, exploration_initial_eps=0.02, tensorboard_log=tensorboard_folder)              #2) Load an existing one from your own files
#print("DQN: I start training now")
model.learn(total_timesteps=1200000, tb_log_name = tensorboard_run_name) #Where the learning happens
#model.save(save_as) #Saving the wisdom for later 

"""
#PPO2-approach

model_name = "PP02_drill_model"
#Chose one of the two lines below (#1 or #2):
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./algorithm_performance_comparison_v0.2/")              #1) Make a new model
#model = PPO2.load(model_name, env, tensorboard_log="./algorithm_performance_comparison/")                  #2) Load an existing one from your own files
print("PPO2: I start training now")
model.learn(total_timesteps=200000, tb_log_name = "PPO2") #Where the learning happens
model.save(model_name) #Saving the wisdom for later 

#A2C-approach

model_name = "A2C_drill_model"
#Chose one of the two lines below (#1 or #2):
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./algorithm_performance_comparison_v0.2/")               #1) Make a new model
#model = A2C.load(model_name, env, tensorboard_log="./algorithm_performance_comparison/")                   #2) Load an existing one from your own files
print("A2C: I start training now")
model.learn(total_timesteps=200000, tb_log_name = "A2C") #Where the learning happens
model.save(model_name) #Saving the wisdom for later 
"""
"""
#ACER-approach [NOT SURE ABOUT THIS ONE]

env = make_vec_env(env_name,n_envs=4) #Problem?
model_name = "ACER_drill_model"
#Chose one of the two lines below (#1 or #2):
model = ACER(MlpPolicy, env, verbose=1, tensorboard_log="./algorithm_performance_comparison_v0.2/")              #1) Make a new model
#model = ACER.load(model_name, env, tensorboard_log="./algorithm_performance_comparison/")                  #2) Load an existing one from your own files
print("I start training now")
model.learn(total_timesteps=200000, tb_log_name = "ACER_1") #Where the learning happens
model.save(model_name) #Saving the wisdom for later 
"""
#ACKTR-approach
"""
env = make_vec_env(env_name,n_envs=4)
model_name = "ACKTR_drill_model"
#Chose one of the two lines below (#1 or #2):
model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="./algorithm_performance_comparison_v0.2/")             #1) Make a new model
#model = ACKTR.load(model_name, env, tensorboard_log="./algorithm_performance_comparison/")                 #2) Load an existing one from your own files
print("I start training now")
model.learn(total_timesteps=200000, tb_log_name = "ACKTR") #Where the learning happens
model.save(model_name) #Saving the wisdom for later 
"""

"""
for episode in range(10):
    	done= False
	steps = 0
	while done==False:
		action = agent.get_action()
		state, reward, done, info = env.step(action)
		env.render()		

	env.display_environment()
"""

print("Im done training and I will show you the results")
#Show the result of the training
obs = env.reset()
for episode in range (10):
	done = False
	while done == False:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		#env.render()
		#print('distance: ',obs[8], '   direction: ', obs[9])
		#env.observation_space_container.display_targets()
		#print(rewards)
		#print(obs)
	env.display_environment()
	state = env.reset()
	
	env.close()
	print('[EPISODE ENDED]')

print("done")
