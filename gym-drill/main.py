import gym
import gym_drill
import random
import numpy as np 

from gym_drill.envs.customAdditions import Coordinate

STARTLOCATION = Coordinate(0.0,0.0)
BIT_INITIALIZATION = [0.0,0.0,0.0]

env_name = 'drill-v0'
env = gym.make(env_name,startLocation = STARTLOCATION, bitInitialization = BIT_INITIALIZATION)

print("Obs space", env.observation_space)
print("action space", env.action_space)

class Agent():
	def __init__(self, env):
		self.action_size = env.action_space.n
		print("Action size", self.action_size)

	def get_action(self):
		action = random.choice(range(self.action_size))
		return action

agent = Agent(env)
state = env.reset()

for episode in range(10):
	done= False
	while done==False:
		action = agent.get_action()
		state, reward, done, info = env.step(action)
		env.render()
	state = env.reset()
	env.close()