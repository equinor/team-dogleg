import gym
import gym_drill
import random

from gym_drill.envs.customAdditions import Coordinate

START_LOCATION = Coordinate(100.0, 380.0)
TARTGET_LOCATION = Coordinate(500,100)
TARGETS = ((Coordinate(500,100),30), (Coordinate(200,100),20))

BIT_INITIALIZATION = [0.0,0.0,0.0]


env_name = 'drill-v0'
env = gym.make(env_name)
env.initParameters(START_LOCATION,TARGETS,BIT_INITIALIZATION)

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
state = env.reset(START_LOCATION,BIT_INITIALIZATION)

for _ in range(2000):
	action = agent.get_action()
	state, reward, done, info = env.step(action)

	env.render()

env.close()