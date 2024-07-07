import sys
from controller import Supervisor
from typing import Any, Tuple, List


try:
    import numpy as np
    import gymnasium as gym
    from gymnasium import Env, spaces
    from gymnasium.envs.registration import EnvSpec, register
except ImportError:
    sys.exit(
        "Please make sure you have all dependencies installed. "
        "Run: `pip install numpy gymnasium stable_baselines3`"
    )


class Pioneer3atEnv(Env):
    def __init__(self, max_episode_steps: int = 1000) -> None:
        super().__init__()

        # Gymnasium generic
        self.theta_threshold_radians: float = 0.2
        self.x_threshold: float = 0.3
        high: np.ndarray = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space: spaces.Discrete = spaces.Discrete(2)
        self.observation_space: spaces.Box = spaces.Box(-high, high, dtype=np.float32)
        self.state: np.ndarray = None
        self.spec: EnvSpec = EnvSpec(
            id="Pioneer3atEnv-v1", max_episode_steps=max_episode_steps
        )

        # Environment specific
        self.__timestep: int = int(self.getBasicTimeStep())
        self.__wheels: List[Any] = []
        self.__pendulum_sensor: Any = None

    def reset(self, seed: Any = None, options: Any = None) -> Tuple[np.ndarray, dict]:
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
            wheel.setVelocity(0.0)
            self.__wheels.append(wheel)

        # Pendulum sensor
        self.__pendulum_sensor = self.getDevice("position sensor")
        self.__pendulum_sensor.enable(self.__timestep)

        # internal state
        super().step(self.__timestep)

        # Gymnasium generic
        self.state = np.array([0, 0, 0, 0], dtype=np.float32)
        info: dict = {}

        return self.state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Execute the action
        for wheel in self.__wheels:
            wheel.setVelocity(1.3 if action == 1 else -1.3)

        # Get the new state
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

        # Check if the episode is done
        done = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        # Reward
        reward = 0 if done else 1

        info = {}

        return self.state.astype(np.float32), reward, done, False, info

    def render(self, mode: str = "human") -> None:
        pass


def wait_for_y():
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# register the environment
register(
    id="Pioneer3atEnv-v1",
    entry_point=Pioneer3atEnv(),
    max_episode_steps=1000,
)
