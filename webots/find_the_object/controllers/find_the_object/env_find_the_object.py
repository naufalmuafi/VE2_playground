"""

finder_controller for finding the object project.

Using Webots and Stable Baselines3.

"""

import sys
from controller import Robot, Display, Supervisor
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


class FTO_Env(Supervisor, Env):
    def __init__(self):
        # Initialize the Robot class
        super().__init__()

        # register the Environment
        self.spec: EnvSpec = EnvSpec(id="FTO-v1")

        # Set the time step
        self.__timestep: int = int(self.getBasicTimeStep())

        # set the speed of robot
        self.speed = 1.3

        # Get the camera device
        self.camera = self.getDevice("camera")
        self.camera.enable(self.__timestep)
        self.camera.recognitionEnable(self.__timestep)
        self.camera.enableRecognitionSegmentation()

        # Get the display device
        self.display = self.getDevice("segmented image display")

        # Get the left and right wheel motors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Set the action and observation spaces
        self.action_space = spaces.Box(
            low=-self.speed, high=self.speed, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.camera.getHeight(), self.camera.getWidth(), 3),
            dtype=np.uint8,
        )

        # Initialize robot and target positions
        self.robot_node = self.getFromDef("ROBOT")
        self.targets = [self.getFromDef("TARGET_1"), self.getFromDef("TARGET_2")]
        self.max_distance = np.sqrt(1.5**2 + 1.5**2)  # Assuming a 1.5x1.5 arena

    def run(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()

        while self.step(self.__timestep) != -1:
            if (
                self.camera.isRecognitionSegmentationEnabled()
                and self.camera.getRecognitionSamplingPeriod() > 0
            ):
                # Get the segmented image and display it in the Display
                data = self.camera.getRecognitionSegmentationImage()
                if data:
                    # print(f"segmented image: {data}")
                    segmented_image = self.display.imageNew(
                        data, Display.BGRA, width, height
                    )
                    self.display.imagePaste(segmented_image, 0, 0, False)
                    self.display.imageDelete(segmented_image)

    def reset(self, seed: Any = None, options: Any = None) -> Tuple[np.ndarray, dict]:
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # reset robot's position and orientation
        self.robot_node.getField("translation").setSFVec3f(
            [-0.5912347141931845, -0.03026246706094393, -0.00023420748536638614]
        )
        self.robot_node.getField("rotation").setSFRotation(
            [-0.014121838568203569, -0.044720364276457, 0.9988997260458302, 3]
        )
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        info = {}

        return self.get_observation(), info

    def get_observation(self) -> np.ndarray:
        # Get camera image as a buffer
        img = self.camera.getImage()
        # Convert the buffer to a numpy array and reshape it
        img = np.frombuffer(img, dtype=np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4)
        )
        # Drop the alpha channel
        img = img[:, :, :3]

        return img

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Apply the action to set the velocity of the motors
        left_speed, right_speed = action
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Step the simulation
        super().step(self.__timestep)

        # Get the new observation
        new_obs = self.get_observation()

        # Calculate the reward
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.is_done()

        truncation = False
        info = {}

        # Return the observation, reward, done, truncation, and info
        return new_obs, reward, done, truncation, info

    def calculate_reward(self) -> float:
        # calculate the distance between the robot and the target
        # robot_position = self.robot_node.getField("translation").getSFVec3f()
        robot_position = self.robot_node.getPosition()
        distances = [
            np.linalg.norm(
                np.array(robot_position[:2]) - np.array(target.getPosition()[:2])
            )
            for target in self.targets
        ]

        # reward function
        if self.is_target_visible():
            reward = -min(distances)
            # reward = 1 - (min(distances) / self.max_distance)
        else:
            reward = -0.1

        return reward

    def is_done(self) -> bool:
        robot_position = self.robot_node.getPosition()
        distances = [
            np.linalg.norm(
                np.array(robot_position[:2]) - np.array(target.getPosition()[:2])
            )
            for target in self.targets
        ]

        done = True if min(distances) < 0.1 else False

        return done

    def is_target_visible(self) -> bool:
        # check if the target is visible in the camera image
        for obj in self.camera.getRecognitionObjects():
            if obj.get_colors() == [0.666667, 0, 0]:  # Target color (red)
                return True
        return False

    def render(self, mode: str = "human") -> None:
        pass


# helper function to wait for user input
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
    id="FTO-v1",
    entry_point=lambda: FTO_Env(),
    max_episode_steps=1000,
)


if __name__ == "__main__":
    env = FTO_Env()
    env.run()
