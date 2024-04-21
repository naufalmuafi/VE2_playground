from stable_baselines3 import A2C

model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log="./tensorboard_log/")

model.learn(total_timesteps=10000, tb_log_name="run_1")

# pass reset_num_timesteps=False to continue the training curve in tensorboard
# by default, it will create a new curve
# keep tb_log_name constant to have continuous curve
model.learn(total_timesteps=10000, tb_log_name="run_2", reset_num_timesteps=False)
model.learn(total_timesteps=10000, tb_log_name="run_3", reset_num_timesteps=False)