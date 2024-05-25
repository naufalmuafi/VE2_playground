'''
OpenAI Gym Integration
at Webots

Reinforcement Learning
with Stable-Baselines3
'''

import sys

try:
  import gymnasium as gym
  from stable_baselines3 import PPO
except ImportError:
  sys.exit(
		'Please make sure you have all dependencies installed!'
	)

# instance the environment
env = gym.make()

# load the model
model = PPO.load('models', env=env)

# run the model in the agent environment
obs = env.reset()[0]

for _ in range(100000):
  action, _states = model.predict(obs)
  obs, reward, done, _, _info = env.step(action)
  print(obs, reward, done)
  
  if done:
    obs = env.reset()[0]