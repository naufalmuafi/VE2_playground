'''
Custom Gym Environment
for Webots Integration

with Pioneer3at Robot
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor # type: ignore
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

# register the module as a gym environment, so once registered, the id is usable in gym.make()
register(
  id='Pioneer3at-v0',
  entry_point='pioneer3at_env:Pioneer3atEnv',
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/

class Pioneer3atEnv(Supervisor, gym.Env):
  # metadata is a required attribute
  # render_modes in our environment is either None or 'human'.  
  metadata = {"render_modes": ["human"]}
  
  def __init__(self, max_episode_steps=1000):
    super().init()