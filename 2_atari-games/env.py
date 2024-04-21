from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)

model = A2C.load("A2C_atari", env=vec_env)

# evaluate agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print("model")
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model_v2 = A2C.load("v2_A2C_atari", env=vec_env)

# evaluate agent
mean_reward, std_reward = evaluate_policy(model_v2, model_v2.get_env(), n_eval_episodes=10)

print("v2_model")
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

obs = vec_env.reset()
while True:
    action, _states = model_v2.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")