import gym
import gym_drill
import random

env_name = 'drill-v0'
env = gym.make(env_name)

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



for _ in range(2000):
	action = agent.get_action()
	state, reward, done, info = env.step(action)

	env.render()

env.close()
