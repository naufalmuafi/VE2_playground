'''
Using Stable-Baselines3 to train The Warehouse Robot
'''

import os
import numpy as np
import gymnasium as gym
import warehouse_robot_env
from stable_baselines3 import A2C, PPO

# train with A2C Algoritm
def train_A2C():
  # where to store trained model and logs
  model_dir = 'models'
  log_dir = 'logs'
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(log_dir, exist_ok=True)
    
  env = gym.make('WarehouseRobot-v0')
    
  # use Advantage Actor-Critic (A2C) algorithm
  # use MLP policy for observation space 1D-vector
  model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    
  TIMESTEPS = 1000
  iters = 0
    
  while True:
    iters += 1
      
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{model_dir}/a2c_{TIMESTEPS * iters}')

# test with A2C algorithm
def test_A2C(render=True):
  env = gym.make('WarehouseRobot-v0', render_mode='human' if render else None)
    
  # load the model
  model = A2C.load('models/a2c_15000', env=env)
    
  # run a test
  obs = env.reset()[0]
  done = False
  while not done:
    action, _states = model.predict(obs, deterministic=True) # deterministic=True means no random action
    obs, _rewards, done, _, _info = env.step(action)
        
    if done:
      break

# unit testing
if __name__ == '__main__':
    # train and test the model with A2C algorithm
    # train_A2C()
    test_A2C()