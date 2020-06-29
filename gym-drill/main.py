import gym
import gym_drill
import random
import numpy as np 

from gym_drill.envs.customAdditions import Coordinate

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
#from stable_baselines.deepq.policies import LnCnnPolicy
#from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

#Setting up the environment
STARTLOCATION = Coordinate(100.0,400.0)
BIT_INITIALIZATION = [3.5*np.pi/4,0.0,0.0]

env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)
model_name = "deepq_gym-drill-two_random_targets_v0.0"

print("Obs space", env.observation_space)
print("action space", env.action_space)


#Using Stable-Baselines to teach an agent 

#Chose one of the two lines below (#1 or #2):
#model = DQN(MlpPolicy, env, verbose=1)		#1) Make a new model
model = DQN.load(model_name, env)			#2) Load an existing one from your own files

model.learn(total_timesteps=10000) #Where the learning happens

model.save(model_name) #Saving the wisdom for later 
del model # removing the model to demonstrate saving and loading
model = DQN.load(model_name, env) #Upload the model you just saved
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

#Show the result of the training
obs = env.reset()
for episode in range (10):
	done = False
	while done == False:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		env.render()
	state = env.reset()
	env.close()