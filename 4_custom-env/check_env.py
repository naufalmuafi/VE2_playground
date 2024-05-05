from stable_baselines3.common.env_checker import check_env
from snake_env import SnakeGameEnv

env = SnakeGameEnv()

# It will check your custom environment and output additional warnings if needed
check_env(env)