import gym
import gym_drill
import random
import numpy as np 
import matplotlib.pyplot as plt
import os

from gym_drill.envs.Coordinate import Coordinate
from gym_drill.envs.Policies import CustomPolicy
from gym_drill.envs import environment_config as cfg

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
warnings.simplefilter(action='ignore', category=Warning)

# Creating an environment with default settings. See register function for details
ENV_name = 'drill-v0'
ENV = gym.make(ENV_name)

# Foldernames
TENSORBOARD_FOLDER_DQN = "./tensorboard_logs/DQN/"

def train_new_DQN(total_timesteps,save_name):
	model = DQN(LnMlpPolicy, ENV, verbose=1, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	model.learn(total_timesteps=total_timesteps, tb_log_name = "DQN")
	save_location = "./trained_models/" + save_name
	model.save(save_location)
	print("Done training with DQN algorithm.")
	print("Results have been saved in ", save_location)

# To load from trained_models folder do: ./trained_models/NAME
def train_existing_DQN(model_to_load,total_timesteps,save_name,*,exploration_initial_eps=0.02,learning_rate= 0.0005):
	print("Loading existing model from ", model_to_load)
	model = DQN.load(model_to_load, ENV, exploration_initial_eps=exploration_initial_eps, learning_rate= learning_rate, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	print("Model loaded and training starts...")
	model.learn(total_timesteps=total_timesteps, tb_log_name = "DQN")
	save_location = "./trained_models/" + save_name
	model.save(save_location)
	print("Done training with DQN algorithm.")
	print("Results have been saved in ", save_location)

def get_trained_model(model_to_load):
	model = DQN.load(model_to_load, ENV, exploration_initial_eps=exploration_initial_eps, learning_rate= learning_rate, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	return model
    	
# Will display model from trained_models folder. To override, specify FOLDERNAME in source_folder
def display_agent(model,*,num_episodes = 5,source_folder = "./trained_models/",vector = False):
	if not vector:
		try:
			model_to_load = source_folder + model 
			trained_model = DQN.load(model_to_load, ENV, tensorboard_log=TENSORBOARD_FOLDER_DQN)
		except Exception as e:
			print("Failed to load model.")
			print("If model is not inside the trained_model folder, override the source_folder to match the desired folder")
			print(str(e))
			os._exit(0)

		# Show the result of the training
		obs = ENV.reset()
		for episode in range (num_episodes):
			done = False
			while not done:
				action, _states = trained_model.predict(obs)
				obs, rewards, done, info = ENV.step(action)
			ENV.display_environment()
			state = ENV.reset()
			
			ENV.close()
			print('[EPISODE ENDED]')

	else:
		print("Vectorized env not implemented yet")

# Change mode to path to get path data
def benchmark_environment(targets,hazards,model,*, 
						startpos=Coordinate(cfg.SCREEN_X*0.1,cfg.SCREEN_Y*0.8), 
						bit_data=[random.uniform(np.pi/2,np.pi),0.0,0.0],
						num_runs = 1,
						mode="display"):
	# Verify mode
	if mode != "display" and mode != "path":
		print("Invalid mode selected!")
		os._exit(0)

	env = gym.make('drill-v0',startLocation = startpos, activate_hazards=True)
	env.load_predefined_env(targets,hazards)
	obs = np.array(env.get_state())

	done = False
	while not done:
		action,states = model.predict(obs)    
		obs, rewards, done, info = env.step(action)
	if mode == "display":
		env.display_environment()
	else:
		path = env.get_path()
	
	env.close()

def main():
	train_new_DQN(10,"final_test")
	display_agent("final_test")
	"""
	#PPO2-approach

	model_to_load = "PPO2_drill_model"
	save_as = "PPO2_drill_model"
	tensorboard_folder = "./algorithm_performance_comparison/"
	tensorboard_run_name = "PP02"
	#Chose one of the two lines below (#1 or #2):
	model = PPO2(MlpPolicy, ENV, verbose=1, tensorboard_log=tensorboard_folder)              #1) Make a new model
	#model = PPO2.load(model_to_load, ENV, tensorboard_log=tensorboard_folder)               #2) Load an existing one from your own files
	print("PPO2: I start training now")
	model.learn(total_timesteps=100, tb_log_name = tensorboard_run_name) #Where the learning happens
	model.save(save_as) #Saving the wisdom for later 

	#A2C-approach

	model_to_load = "A2C_drill_model"
	save_as = "A2C_drill_model"
	tensorboard_folder = "./algorithm_performance_comparison/"
	tensorboard_run_name = "A2C"
	ENV = make_vec_ENV(ENV_name,n_ENVs=4)
	#Chose one of the two lines below (#1 or #2):
	model = A2C(MlpPolicy, ENV, verbose=1, tensorboard_log=tensorboard_folder)              #1) Make a new model
	#model = A2C.load(model_to_load, ENV, tensorboard_log=tensorboard_folder)               #2) Load an existing one from your own files
	print("A2C: I start training now")
	model.learn(total_timesteps=100, tb_log_name = tensorboard_run_name) #Where the learning happens
	model.save(save_as) #Saving the wisdom for later 

	#ACER-approach

	model_to_load = "ACER_drill_model"
	save_as = "ACER_drill_model"
	tensorboard_folder = "./algorithm_performance_comparison/"
	tensorboard_run_name = "ACER"
	ENV = make_vec_ENV(ENV_name,n_ENVs=4)
	#Chose one of the two lines below (#1 or #2):
	model = ACER(MlpPolicy, ENV, verbose=1, tensorboard_log=tensorboard_folder)              #1) Make a new model
	#model = ACER.load(model_to_load, ENV, tensorboard_log=tensorboard_folder)               #2) Load an existing one from your own files
	print("ACER: I start training now")
	model.learn(total_timesteps=100, tb_log_name = tensorboard_run_name) #Where the learning happens
	model.save(save_as) #Saving the wisdom for later 

	#ACKTR-approach

	model_to_load = "ACKTR_drill_model"
	save_as = "ACKTR_drill_model"
	tensorboard_folder = "./algorithm_performance_comparison/"
	tensorboard_run_name = "ACKTR"
	ENV = make_vec_ENV(ENV_name,n_ENVs=4)
	#Chose one of the two lines below (#1 or #2):
	model = ACKTR(MlpPolicy, ENV, verbose=1, tensorboard_log=tensorboard_folder)              	#1) Make a new model
	#model = ACKTR.load(model_to_load, ENV, tensorboard_log=tensorboard_folder)                 #2) Load an existing one from your own files
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
	model = TRPO(MlpPolicy, ENV, verbose=0, tensorboard_log=tensorboard_folder)              	#1) Make a new model
	#model = TRPO.load(model_to_load, ENV, tensorboard_log=tensorboard_folder)                 #2) Load an existing one from your own files
	print("TRPO: I start training now")
	model.learn(total_timesteps=2000000, tb_log_name = tensorboard_run_name) #Where the learning happens
	model.save(save_as) #Saving the wisdom for later 
	"""

if __name__ == '__main__':
	main()