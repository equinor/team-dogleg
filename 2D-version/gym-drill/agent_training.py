import gym
import gym_drill
import random
import numpy as np 
import os
from random import uniform
#import matplotlib as mpl # To remove plotting in the browser remove this line
#mpl.use("WebAgg") # and remove this line
import matplotlib.pyplot as plt

from gym_drill.envs.Coordinate import Coordinate
#from gym_drill.envs.Policies import CustomPolicy
from gym_drill.envs import environment_config as cfg

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
warnings.simplefilter(action='ignore', category=Warning)
print("-------------------------------------------------------------------------")

# Creating an environment with default settings. See register function for details
ENV_name = 'drill-v0'
ENV = gym.make(ENV_name, bitInitialization= [uniform(0,2*np.pi),uniform(0,np.pi/4),0.0,0.0,0.0,0.0], activate_hazards = True)

#Custom network architecture 
policy_kwargs= dict(act_fun=tf.nn.relu,  layers=[64,64,64,64,64,32,16]) # Use as argument when doing model =...(policy_kwargs = policy_kwargs)


# Foldernames
TRAINED_MODEL_FOLDER_DOCKER = "app/trained_models/"
TRAINED_MODEL_FOLDER_LOCAL = "../trained_models/"

TENSORBOARD_FOLDER_DQN = "../tensorboard_logs/DQN/"
TENSORBOARD_FOLDER_PPO2 = "../tensorboard_logs/PPO2/"

def train_new_DQN(total_timesteps,save_name):
	print("Starting DQN training session of",total_timesteps,"timesteps...")
	model = DQN(LnMlpPolicy, ENV, verbose=1, learning_rate=0.0003,gamma=0.999, exploration_fraction=0.2,policy_kwargs=policy_kwargs,exploration_final_eps=0.0,tensorboard_log=TENSORBOARD_FOLDER_DQN)
	model.learn(total_timesteps=total_timesteps, tb_log_name = "DQN")
	print("Done training with DQN algorithm.")
	save_model(model,save_name)

# To load from trained_models folder do: ./trained_models/NAME
def train_existing_DQN(model_to_load,total_timesteps,save_name,*,exploration_initial_eps=0.0,learning_rate= 0.0003):
	model = get_trained_DQN_model(model_to_load)
	print("Model loaded and training starts...")
	model.learn(total_timesteps=total_timesteps, tb_log_name = "DQN")
	print("Done training with DQN algorithm.")
	save_model(model,save_name)

def get_trained_DQN_model(model_to_load,*,exploration_initial_eps=0.02,learning_rate= 0.0005):
	load_location = TRAINED_MODEL_FOLDER_DOCKER + model_to_load
	print("Loading existing model from ", load_location)
	try:
		model = DQN.load(load_location, ENV, exploration_initial_eps=exploration_initial_eps, learning_rate= learning_rate, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	except Exception:
		print(load_location, "not found.")
		load_location = TRAINED_MODEL_FOLDER_LOCAL + model_to_load
		print("Assuming you are running locally and will load from",load_location)
		model = DQN.load(load_location, ENV, exploration_initial_eps=exploration_initial_eps, learning_rate= learning_rate, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	
	return model

def train_new_PPO2(total_timesteps,save_name):
	model = PPO2(MlpPolicy, ENV, verbose=1, tensorboard_log=TENSORBOARD_FOLDER_PPO2)
	model.learn(total_timesteps=total_timesteps, tb_log_name = "PPO2")
	print("Done training with PPO2 algorithm.")
	save_model(model,save_name)

# To load from trained_models folder do: ./trained_models/NAME
def train_existing_PPO2(model_to_load,total_timesteps,save_name):
	print("Loading existing model from ", load_location)
	model = get_trained_PPO2_model(model_to_load) 
	print("Model loaded and training starts...")
	model.learn(total_timesteps=total_timesteps, tb_log_name = "PPO2")
	print("Done training with PPO2 algorithm.")
	save_model(model,save_location)

def get_trained_PPO2_model(model_to_load):
	load_location = TRAINED_MODEL_FOLDER_DOCKER + model_to_load
	try:
		model = PPO2.load(load_location, ENV, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	except Exception:
		load_location = TRAINED_MODEL_FOLDER_LOCAL + model_to_load
		model = PPO2.load(load_location, ENV, tensorboard_log=TENSORBOARD_FOLDER_DQN)
	
	return model
    	
def save_model(model,save_name,*,folder_name = TRAINED_MODEL_FOLDER_DOCKER):
	save_location = folder_name + save_name
	try:
		model.save(save_location)
	except FileNotFoundError:
		# We are not running from Docker.
		save_location = TRAINED_MODEL_FOLDER_LOCAL + save_name
		model.save(save_location)

	print("Results have been saved in ", save_location)

ENV_DISP = gym.make(ENV_name, bitInitialization= [uniform(0,2*np.pi),uniform(0,np.pi/4),0.0,0.0,0.0,0.0], activate_hazards = True,load = False)
# Will display model from trained_models folder. To override, specify FOLDERNAME in source_folder
def display_agent(model,*,num_episodes = 1,source_folder = TRAINED_MODEL_FOLDER_DOCKER,vector = False):
	if not vector:
		try:
			model_to_load = source_folder + model 
			trained_model = DQN.load(model_to_load, ENV_DISP)
		except Exception as e:
			try:
				source_folder = TRAINED_MODEL_FOLDER_LOCAL
				model_to_load = source_folder + model
				trained_model = DQN.load(model_to_load, ENV_DISP)
			except Exception as e:
				print("Failed to load model.")
				print("If model is not inside the trained_model folder, override the source_folder to match the desired folder")
				print(str(e))
				os._exit(0)

		# Show the result of the training
		obs = ENV_DISP.reset()
		for episode in range (num_episodes):
			done = False
			while not done:
				action, _states = trained_model.predict(obs)
				obs, rewards, done, info = ENV_DISP.step(action)

			fig_xy = ENV_DISP.get_xy_plane_figure()
			fig_xz = ENV_DISP.get_xz_plane_figure()
			fig_3d = ENV_DISP.get_3d_figure()
			print('[EPISODE ENDED]')
			plt.show()	

			obs = ENV_DISP.reset()			

	else:
		print("Vectorized env not implemented yet")
def get_environment_figures(model,*,source_folder = TRAINED_MODEL_FOLDER_DOCKER,vector = False):
	if not vector:
		try:
			model_to_load = source_folder + model 
			trained_model = DQN.load(model_to_load, ENV_DISP)
		except Exception as e:
			try:
				source_folder = TRAINED_MODEL_FOLDER_LOCAL
				model_to_load = source_folder + model
				trained_model = DQN.load(model_to_load, ENV_DISP)
			except Exception as e:
				print("Failed to load model.")
				print("If model is not inside the trained_model folder, override the source_folder to match the desired folder")
				print(str(e))
				os._exit(0)

		# Show the result of the training
		obs = ENV_DISP.reset()
		for episode in range (1):
			done = False
			while not done:
				action, _states = trained_model.predict(obs)
				obs, rewards, done, info = ENV_DISP.step(action)

			fig_xy = ENV_DISP.get_xy_plane_figure()
			fig_xz = ENV_DISP.get_xz_plane_figure()
			fig_3d = ENV_DISP.get_3d_figure()
			return fig_xy,fig_xz,fig_3d
	else:
		print("Vectorized env not implemented yet")


# Change mode to path to get path data
def benchmark_environment(targets,hazards,model,*, 
						startpos=Coordinate(cfg.SCREEN_X*0.1,cfg.SCREEN_Y*0.8,0), 
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

if __name__ == '__main__':
	print("You are running this specifc file!")