"""

finder_controller for finding the object project.

Using Webots and Stable Baselines3.

"""

import sys
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
    def __init__(self):
        # Initialize the Robot class
        super().__init__()

        # register the Environment
        self.spec: EnvSpec = EnvSpec(id="FTO-v1")

        # Set the time step
        self.__timestep: int = int(self.getBasicTimeStep())

        # Initialize the Robot class
        random.seed(42)

        # set the speed of the motors
        self.speed = 1.3

        # set the threshold of the target area
        self.threshold = 0.35

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
        self.targets = self.getFromDef("TARGET_1")
        self.max_distance = np.sqrt(1.5**2 + 1.5**2)  # Assuming a 1.5x1.5 arena

        # Initialize the target area
        self.previous_target_area = 0.0

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
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        # set the initial velocity of the motors randomly
        initial_direction = random.choice([-1, 1]) * self.speed / 1.5
        self.left_motor.setVelocity(-initial_direction)
        self.right_motor.setVelocity(initial_direction)

        # gymnasium generic variables
        self.state = np.zeros(
            (self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8
        )
        info = {}

        return self.state, info

    def step(self, action: int):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        # action

        # Get the new state
        super().step(self.__timestep)

        # observation

        if (
            self.camera.isRecognitionSegmentationEnabled()
            and self.camera.getRecognitionSamplingPeriod() > 0
        ):
            objects = self.camera.getRecognitionObjects()
            data = self.camera.getRecognitionSegmentationImage()

            if data:
                self.display_segmented_image(data, width, height)

            # calculate the target area
            target_area = self.calculate_target_area(data, width, height, frame_area)

        # termination
        done = self.is_done(target_area, self.threshold)

        # reward
        reward = 0

        info = {}

        return self.state.astype(np.int32), reward, done, False, info

    def render(self, mode: str = "human") -> None:
        pass

    def is_done(self, target_area, threshold=0.25):
        done = False
        if target_area >= threshold:
            print(f"Target area meets or exceeds {threshold * 100:.2f}% of the frame.")
            self.stop_motors()
            done = True

        return done

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
