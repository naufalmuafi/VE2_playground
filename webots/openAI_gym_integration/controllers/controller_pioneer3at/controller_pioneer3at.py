import os
import env
import gymnasium as gym
from stable_baselines3 import PPO


TIMESTEPS = 1000

# where to store trained model and logs
path: str = {"model": "models", "log": "logs"}

# create the folder from path
os.makedirs(path["model"], exist_ok=True)
os.makedirs(path["log"], exist_ok=True)


def train_PPO(model_dir: str, log_dir: str, timesteps: int) -> None:
    # define the environment
    env = gym.make("Pioneer3at-v1")

    # use Proximal Policy Optimization (PPO) algorithm
    # use MLP policy for observation space 1D-vector
    print("Training the model with PPO...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # train and save the model
    model.learn(total_timesteps=timesteps)
    model.save(f"{model_dir}/ppo_{timesteps}")


def test_PPO(model_dir: str, timesteps: int = TIMESTEPS) -> None:
    # define the environment
    env = gym.make("Pioneer3at-v1")

    # load the model
    model = PPO.load(f"{model_dir}/ppo_{timesteps}", env=env)
    print("Load Model Successful") if model else print("Failed to Load Model")

    # run a test
    obs, _ = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)

        if done:
            obs, _ = env.reset()
            print("Test Terminated.")

        print("Test Successful") if i == timesteps else None


def wait_for_y() -> None:
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # train and test the model with A2C algorithm
    train_PPO(path["model"], path["log"], TIMESTEPS)

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("Test the Environment with Predicted Value")

    test_PPO(path["model"], TIMESTEPS)
