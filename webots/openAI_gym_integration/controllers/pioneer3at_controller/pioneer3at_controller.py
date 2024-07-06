"""
Using Stable-Baselines3 to train The Pioneer3AT Robot
"""

import os
import gymnasium as gym
import pioneer3at_controller_env
from stable_baselines3 import PPO


def train_PPO():
    # where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("Pioneer3atEnv-v0")

    # use Proximal Policy Optimization (PPO) algorithm
    # use MLP policy for observation space 1D-vector
    TIMESTEPS = 1000

    print("Training the model...")
    model = PPO("MlpPolicy", env, n_steps=2048, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"{model_dir}/ppo_{TIMESTEPS}")


def test_PPO(render=True):
    env = gym.make("Pioneer3atEnv-v0")

    # load the model
    TIMESTEPS = 1000
    model = PPO.load(f"models/ppo_{TIMESTEPS}", env=env)

    # run a test
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)
        if done:
            obs, _ = env.reset()


def wait_for_y():
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# unit testing
if __name__ == "__main__":
    # train and test the model with A2C algorithm
    train_PPO()

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("OK!")

    test_PPO()
