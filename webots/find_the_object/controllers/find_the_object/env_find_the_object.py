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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.camera.getHeight(), self.camera.getWidth(), 3),
            dtype=np.uint8,
        )

        # Initialize robot and target positions
        self.robot_node = self.getDevice("MyBot")
        self.targets = [self.getDevice("target_1"), self.getDevice("target_2")]
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
