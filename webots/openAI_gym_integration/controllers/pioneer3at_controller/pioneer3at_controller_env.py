import sys
from controller import Supervisor
from gymnasium.envs.registration import register

try:
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        "Please make sure you have all dependencies installed. "
        'Run: "pip install numpy gymnasium stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # Gymnasium generic
        self.theta_threshold_radians = 0.2
        self.x_threshold = 0.3
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(
            id="Pioneer3atEnv-v0", max_episode_steps=max_episode_steps
        )

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__pendulum_sensor = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord("Y"):
            super().step(self.__timestep)

    def reset(self, seed=None, options=None):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Motors
        self.__wheels = []
        for name in [
            "back left wheel",
            "back right wheel",
            "front left wheel",
            "front right wheel",
        ]:
            wheel = self.getDevice(name)
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        # Sensors
        self.__pendulum_sensor = self.getDevice("position sensor")
        self.__pendulum_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        # Gymnasium generic
        self.state = np.array([0, 0, 0, 0]).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Execute the action
        for wheel in self.__wheels:
            wheel.setVelocity(1.3 if action == 1 else -1.3)
        super().step(self.__timestep)

        # Observation
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        self.state = np.array(
            [
                robot.getPosition()[0],
                robot.getVelocity()[0],
                self.__pendulum_sensor.getValue(),
                endpoint.getVelocity()[4],
            ]
        )

        # Done
        done = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        # Reward
        reward = 0 if done else 1

        return self.state.astype(np.float32), reward, done, False, {}

    def render(self, mode="human"):
        pass


# Register the environment
def make_env():
    return OpenAIGymEnvironment()


def wait_for_y():
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


register(
    id="Pioneer3atEnv-v0",
    entry_point=make_env,
    max_episode_steps=1000,
)


def main():
    # Initialize the environment
    env = gym.make("Pioneer3atEnv-v0")

    print(f"Check the environment: {env}...")
    check_env(env)

    # Train
    print("Training the model...")
    model = PPO("MlpPolicy", env, n_steps=2048, verbose=1)
    model.learn(total_timesteps=1e5)

    # Replay
    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("OK!")

    print("Replaying the model...")
    obs, _ = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)
        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
