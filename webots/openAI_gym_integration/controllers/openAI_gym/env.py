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
    
    # Open AI Gym generic
    self.theta_threshold_radians = 0.2
    self.x_threshold = 0.3
    
    high = np.array(
      [
        self.x_threshold * 2,
        np.finfo(np.float32).max,
        self.theta_threshold_radians * 2,
        np.finfo(np.float32).max
      ],
      dtype=np.float32
    )
    
    self.action_space = gym.spaces.Discrete(2)
    self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
    
    self.state = None
    self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

    # Environment specific
    self.__timestep = int(self.getBasicTimeStep())
    self.__wheels = []
    self.__pendulum_sensor = None

    # Tools
    self.keyboard = self.getKeyboard()
    self.keyboard.enable(self.__timestep)
  
  def reset(self, seed=None):
    super().reset(seed=seed) # required to control the randomness and reproduce the scenario
    
    # reset the simulation
    self.simulationResetPhysics()
    self.simulationReset()
    super().step(self.__timestep)
    
    # Intialize motor devices
    self.__wheels = []
    for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
      wheel = self.getDevice(name)
      wheel.setPosition(float('inf'))
      wheel.setVelocity(0)
      self.__wheels.append(wheel)

    # Sensors
    self.__pendulum_sensor = self.getDevice('position sensor')
    self.__pendulum_sensor.enable(self.__timestep)

    # Internals
    super().step(self.__timestep)

    # Open AI Gym generic
    return np.array([0, 0, 0, 0]).astype(np.float32)