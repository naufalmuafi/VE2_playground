# """

# finder_controller for finding the object project.

# Using Webots and Stable Baselines3.

# """

import sys
from enum import Enum
from controller import Supervisor, Display
from typing import Any, Tuple, List

try:
    import random
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
    def __init__(self, max_episode_steps: int = 1000) -> None:
        # Initialize the Robot class
        super().__init__()
        random.seed(42)

        # register the Environment
        self.spec: EnvSpec = EnvSpec(id="FTO-v1", max_episode_steps=max_episode_steps)

        # set the max_speed of the motors
        self.max_speed = 1.3

        # set the threshold of the target area
        self.target_threshold = 0.35

        # get the camera devices
        self.camera = self.getDevice("camera")

        # Set the action spaces: 0 = left, 1 = right, 2 = forward, 3 = stop
        # case 1: discrete
        # self.action_space: spaces.Discrete = spaces.Discrete(len(RobotAction))

        # case 2: continuous
        self.action_space = spaces.Box(
            low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32
        )

        # Set the observation spaces
        # RGB images of shape (camera_width, camera_height, 3) and a target value in range 0-100
        # self.observation_space = spaces.Dict(
        #     {
        #         "segmentation_img": spaces.Box(
        #             low=0,
        #             high=255,
        #             shape=(self.camera.getHeight(), self.camera.getWidth(), 3),
        #             dtype=np.uint8,
        #         ),
        #         "target_area": spaces.Box(
        #             low=0, high=100, shape=(1,), dtype=np.float32
        #         ),
        #     }
        # )

        # pure image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.camera.getHeight(), self.camera.getWidth(), 3),
            dtype=np.uint8,
        )

        # environment specification
        self.state = None
        self.display: Any = None
        self.motors: List[Any] = []
        self.__timestep: int = int(self.getBasicTimeStep())

    def reset(self, seed: Any = None, options: Any = None):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # get the camera devices
        self.camera = self.getDevice("camera")
        self.camera.enable(self.__timestep)
        self.camera.recognitionEnable(self.__timestep)
        self.camera.enableRecognitionSegmentation()

        # Get the display device
        self.display = self.getDevice("segmented image display")

        # Get the left and right wheel motors
        self.motors = []
        for name in ["left wheel motor", "right wheel motor"]:
            motor = self.getDevice(name)
            self.motors.append(motor)
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        # internal state
        super().step(self.__timestep)

        # initial state

        # initial_obs1 = np.zeros(
        #     (self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8
        # )
        # initial_obs2 = np.zeros((1,), dtype=np.float32)
        # self.state = {"segmentation_img": initial_obs1, "target_area": initial_obs2}

        self.state = np.zeros(
            (self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8
        )

        # info
        info: dict = {}

        return self.state, info

    def step(self, action):
        # perform a continuous action
        self.motors[0].setVelocity(action[0])
        self.motors[1].setVelocity(action[1])

        # Get the new state
        super().step(self.__timestep)

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        red = 0
        green = 0
        blue = 0

        # Create a 2D list to store pixel values
        pixels = []

        # initialize data as an array
        data = np.zeros((height, width, 4), dtype=np.uint8)
        target_area = 0  # initialize target_area

        if (
            self.camera.isRecognitionSegmentationEnabled()
            and self.camera.getRecognitionSamplingPeriod() > 0
        ):
            image = self.camera.getImage()
            objects = self.camera.getRecognitionObjects()
            data = self.camera.getRecognitionSegmentationImage()

            if data:
                # Loop through each pixel in the image
                for j in range(height):
                    row = []
                    for i in range(width):
                        # Get the RGB values for the pixel (i, j)
                        red = self.camera.imageGetRed(image, width, i, j)
                        green = self.camera.imageGetGreen(image, width, i, j)
                        blue = self.camera.imageGetBlue(image, width, i, j)

                        # Append the RGB values as a tuple to the row
                        row.append((red, green, blue))

                    # Append the row to the pixels list
                    pixels.append(row)

                self.display_segmented_image(data, width, height)

                # calculate the target area
                target_area = self.calculate_target_area_px(
                    pixels, width, height, frame_area
                )

        # observation
        # self.state = {
        #     "segmentation_img": data[:, :, :3],
        #     "target_area": np.array([target_area], dtype=np.float32),
        # }

        self.state = np.array(pixels, dtype=np.uint8)

        # check if the episode is done
        done = bool(target_area >= self.target_threshold)

        # reward
        reward = target_area if target_area > 0 else 0

        if done:
            reward += 1000

        # info
        info = {}

        return self.state, reward, done, False, info

    def render(self, mode: str = "human") -> None:
        pass

    def display_segmented_image(self, data, width, height):
        segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
        self.display.imagePaste(segmented_image, 0, 0, False)
        self.display.imageDelete(segmented_image)

    def calculate_target_area(self, data, width, height, frame_area):
        target_px = 0

        for y in range(height):
            for x in range(width):
                index = (y * width + x) * 4
                b, g, r, a = data[index : index + 4]

                if r == 170 and g == 0 and b == 0:
                    target_px += 1

        target_area = target_px / frame_area

        return target_area

    def calculate_target_area_px(self, pixels, width, height, frame_area):
        target_px = 0

        for y in range(height):
            for x in range(width):
                # Get the RGB values for the pixel (x, y)
                r, g, b = pixels[y][x]

                # Check if the pixel matches the target color
                if r == 170 and g == 0 and b == 0:
                    target_px += 1

        target_area = target_px / frame_area

        return target_area


# register the environment
register(
    id="FTO-v1",
    entry_point=lambda: FTO_Env(),
    max_episode_steps=1000,
)
