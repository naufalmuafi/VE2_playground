'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import warehouse_robot as wr
import numpy as np

# register the module as a gym environment, so once registered, the id is usable in gym.make()
register(
  id='WarehouseRobot-v0',
  entry_point='warehouse_robot:WarehouseRobotEnv',
)

