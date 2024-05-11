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

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/

class WarehouseRobotEnv(gym.Env):
  # metadata is a required attribute
  # render_modes in our environment is either None or 'human'.
  # render_fps is not used in our env, but we are require to declare a non-zero value.
  metadata = {"render_modes": ["human"], 'render_fps': 4}
  
  def __init__(self, grid_rows=4, grid_cols=5, render_mode=None):
    self.grid_rows = grid_rows
    self.grid_cols = grid_cols
    self.render_mode = render_mode
    
    # initialize the warehouse problem
    self.warehouse_robot = wr.WarehouseRobot(grid_rows, grid_cols, self.metadata['render_fps'])
    
    # define the action space i.e. the possible actions the agent/the robot's can take
    self.action_space = spaces.Discrete(len(wr.RobotAction))
    
    # define the observation space i.e. the possible states/posiitons the agent/the robot can be in
    # used to validate the observation returned by reset() and step()
    # use 1D-vector to represent the robot's position: : [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
    self.observation_space = spaces.Box(
      low=0, 
      high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1]), 
      shape=(4, ), 
      dtype=np.int32
    )
    
  def reset(self, seed=None, options=None):
    super().reset(seed=seed) # required to control the randomness and reproduce the scenario
    
    # Reset the WarehouseRobot. Optionally, pass in seed control randomness and reproduce scenarios.
    self.warehouse_robot.reset(seed=seed)
    
    # construct the obs. state:
    # [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
    obs = np.concatenate((self.warehouse_robot.robot_pos, self.warehouse_robot.target_pos))
    
    # additional info to the return, for debugging purposes
    info = {}
    
    # render environment
    if (self.render_mode=='human'):
      self.render()
    
    # return obsservation and info
    return obs, info