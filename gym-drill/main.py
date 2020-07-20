import gym
import gym_drill
import random
import numpy as np 
import matplotlib.pyplot as plt

from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Policies import CustomPolicy

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy as DQN_MlpPolicy
from stable_baselines.deepq.policies import LnMlpPolicy 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, PPO2, A2C, ACER, ACKTR, TRPO

# Ignore the crazy amount of warnings
import warnings
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Setting up the environment
STARTLOCATION = Coordinate(100,900.0)
BIT_INITIALIZATION = [random.uniform(np.pi/2,np.pi),0.0,0.0] #initial heading is also set to random in the reset function (drill_env.py)


env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION, activate_hazards=True)

print("Obs space", env.observation_space)
print("action space", env.action_space)

#Using Stable-Baselines to teach an agent 

#DQN-approach

model_to_load = "DQN_drill_model_hazards"
save_as = "DQN_drill_model_hazards"
tensorboard_folder = "./algorithm_performance_hazards/"
tensorboard_run_name = "DQN"
#Chose one of the two lines below (#1 or #2):
model = DQN(LnMlpPolicy, env, verbose=1, tensorboard_log=tensorboard_folder)           #1) Make a new model
#model = DQN.load(model_to_load, env, exploration_initial_eps=0.02, learning_rate= 0.0005, tensorboard_log=tensorboard_folder)              #2) Load an existing one from your own files
print("DQN: I start training now")
model.learn(total_timesteps=1000000, tb_log_name = tensorboard_run_name) #Where the learning happens
model.save(save_as) #Saving the wisdom for later 

"""
#PPO2-approach

#PPO2-approach
"""
model_to_load = "PPO2_drill_model"
save_as = "PPO2_drill_model"
tensorboard_folder = "./algorithm_performance_comparison/"
tensorboard_run_name = "PP02"
#Chose one of the two lines below (#1 or #2):
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_folder)              #1) Make a new model
#model = PPO2.load(model_to_load, env, tensorboard_log=tensorboard_folder)               #2) Load an existing one from your own files
print("PPO2: I start training now")
model.learn(total_timesteps=1000000, tb_log_name = tensorboard_run_name) #Where the learning happens
model.save(save_as) #Saving the wisdom for later 

#A2C-approach

model_to_load = "A2C_drill_model"
save_as = "A2C_drill_model"
tensorboard_folder = "./algorithm_performance_comparison/"
tensorboard_run_name = "A2C"
print("Making vectorized env")

env = make_vec_env(env_name,n_envs=4)
#Chose one of the two lines below (#1 or #2):
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_folder)              #1) Make a new model
#model = A2C.load(model_to_load, env, tensorboard_log=tensorboard_folder)               #2) Load an existing one from your own files
print("A2C: I start training now")
model.learn(total_timesteps=100, tb_log_name = tensorboard_run_name) #Where the learning happens
model.save(save_as) #Saving the wisdom for later 
"""
#ACER-approach
"""
model_to_load = "ACER_drill_model"
save_as = "ACER_drill_model"
tensorboard_folder = "./algorithm_performance_comparison/"
tensorboard_run_name = "ACER"
env = make_vec_env(env_name,n_envs=4)
#Chose one of the two lines below (#1 or #2):
model = ACER(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_folder)              #1) Make a new model
#model = ACER.load(model_to_load, env, tensorboard_log=tensorboard_folder)               #2) Load an existing one from your own files
print("ACER: I start training now")
model.learn(total_timesteps=100, tb_log_name = tensorboard_run_name) #Where the learning happens
model.save(save_as) #Saving the wisdom for later 

#ACKTR-approach

model_to_load = "ACKTR_drill_model"
save_as = "ACKTR_drill_model"
tensorboard_folder = "./algorithm_performance_comparison/"
tensorboard_run_name = "ACKTR"
env = make_vec_env(env_name,n_envs=4)
#Chose one of the two lines below (#1 or #2):
model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_folder)              	#1) Make a new model
#model = ACKTR.load(model_to_load, env, tensorboard_log=tensorboard_folder)                 #2) Load an existing one from your own files
print("ACKTR: I start training now")
model.learn(total_timesteps=100, tb_log_name = tensorboard_run_name) #Where the learning happens
model.save(save_as) #Saving the wisdom for later 
"""
#TRPO-approach 
"""
model_to_load = "TRPO_drill_model_1000"
save_as = "TRPO_drill_model_1000"
tensorboard_folder = "./algorithm_performance_comparison_1000/"
tensorboard_run_name = "TRPO"
#Chose one of the two lines below (#1 or #2):
model = TRPO(MlpPolicy, env, verbose=0, tensorboard_log=tensorboard_folder)              	#1) Make a new model
#model = TRPO.load(model_to_load, env, tensorboard_log=tensorboard_folder)                 #2) Load an existing one from your own files
print("TRPO: I start training now")
model.learn(total_timesteps=2000000, tb_log_name = tensorboard_run_name) #Where the learning happens
model.save(save_as) #Saving the wisdom for later 
"""
print("Im done training and I will show you the results")
#Show the result of the training
obs = env.reset()
for episode in range (10):
	done = False
	while done == False:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
	env.display_environment()
	state = env.reset()
	
	env.close()
	print('[EPISODE ENDED]')

print("done")
