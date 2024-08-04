# """

# finder_controller for finding the object project.

# Using Webots and Stable Baselines3.

# """

# import sys
# from enum import Enum
# from controller import Supervisor, Display
# from typing import Any, Tuple, List

# try:
#     import random
#     import numpy as np
#     import gymnasium as gym
#     from gymnasium import Env, spaces
#     from gymnasium.envs.registration import EnvSpec, register
# except ImportError:
#     sys.exit(
#         "Please make sure you have all dependencies installed. "
#         "Run: `pip install numpy gymnasium stable_baselines3`"
#     )


# # Robot action to go in a certain direction in discrete space
# class RobotAction(Enum):
#     LEFT = 0
#     RIGHT = 1
#     FORWARD = 2
#     STOP = 3
#     TURN_OVER = 4


# class FTO_Env(Supervisor, Env):

#     def __init__(self, max_episode_steps: int = 1000) -> None:
#         # Initialize the Robot class
#         super().__init__()
#         random.seed(42)

#         # register the Environment
#         self.spec: EnvSpec = EnvSpec(id="FTO-v1", max_episode_steps=max_episode_steps)

#         # set the max_speed of the motors
#         self.max_speed = 1.3

#         # set the threshold of the target area
#         self.target_threshold = 0.35

#         # Set the action spaces: 0 = left, 1 = right, 2 = forward, 3 = stop
#         # case 1: discrete
#         # self.action_space: spaces.Discrete = spaces.Discrete(len(RobotAction))

#         # case 2: continuous
#         self.action_space = spaces.Box(
#             low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32
#         )

#         # Set the observation spaces
#         # RGB images of shape (camera_width, camera_height, 3) and a target value in range 0-100
#         self.observation_space = spaces.Dict(
#             {
#                 "segmentation_img": spaces.Box(
#                     low=0,
#                     high=255,
#                     shape=(self.camera.getWidth(), self.camera.getHeight(), 3),
#                     dtype=np.uint8,
#                 ),
#                 "target_area": spaces.Box(
#                     low=0, high=100, shape=(1,), dtype=np.float32
#                 ),
#             }
#         )

#         # pure image
#         # self.observation_space = spaces.Box(
#         #     low=0,
#         #     high=255,
#         #     shape=(self.camera.getHeight(), self.camera.getWidth(), 3),
#         #     dtype=np.uint8,
#         # )

#         # environment specification
#         self.state = None
#         self.camera: Any = None
#         self.display: Any = None
#         self.motors: List[Any] = []
#         self.__timestep: int = int(self.getBasicTimeStep())

#     def reset(self, seed: Any = None, options: Any = None):
#         # Reset the simulation
#         self.simulationResetPhysics()
#         self.simulationReset()
#         super().step(self.__timestep)

#         # get the camera devices
#         self.camera = self.getDevice("camera")
#         self.camera.enable(self.__timestep)
#         self.camera.recognitionEnable(self.__timestep)
#         self.camera.enableRecognitionSegmentation()

#         # Get the display device
#         self.display = self.getDevice("segmented image display")

#         # Get the left and right wheel motors
#         self.motors = []
#         for name in ["left wheel motor", "right wheel motor"]:
#             motor = self.getDevice(name)
#             self.motors.append(motor)
#             motor.setPosition(float("inf"))
#             motor.setVelocity(0.0)

#         # internal state
#         super().step(self.__timestep)

#         # initial state
#         initial_obs1 = np.zeros(
#             (self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8
#         )
#         initial_obs2 = 0
#         self.state = {"segmentation_img": initial_obs1, "target_area": initial_obs2}

#         # info
#         info: dict = {}

#         return self.state, info

#     def step(self, action: int):
#         # perform a continuous action
#         self.motors[0].setVelocity(action[0])
#         self.motors[1].setVelocity(action[1])

#         # Get the new state
#         super().step(self.__timestep)

#         width = self.camera.getWidth()
#         height = self.camera.getHeight()
#         frame_area = width * height

#         if (
#             self.camera.isRecognitionSegmentationEnabled()
#             and self.camera.getRecognitionSamplingPeriod() > 0
#         ):
#             data = self.camera.getRecognitionSegmentationImage()

#             if data:
#                 self.display_segmented_image(data, width, height)

#                 # calculate the target area
#                 target_area = self.calculate_target_area(
#                     data, width, height, frame_area
#                 )

#         # observation
#         self.state = {"segmentation_img": data, "target_area": target_area}

#         # check if the episode is done
#         done = bool(self.state["target_area"] >= self.target_threshold)

#         # reward
#         if target_area > 0:
#             reward = target_area
#             if done:
#                 reward += 1000
#         else:
#             reward = 0

#         # info
#         info = {}

#         return self.state.astype(np.int32), reward, done, False, info

#     def render(self, mode: str = "human") -> None:
#         pass

#     def display_segmented_image(self, data, width, height):
#         segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
#         self.display.imagePaste(segmented_image, 0, 0, False)
#         self.display.imageDelete(segmented_image)

#     def calculate_target_area(self, data, width, height, frame_area):
#         target_px = 0

#         for y in range(height):
#             for x in range(width):
#                 index = (y * width + x) * 4
#                 b, g, r, a = data[index : index + 4]

#                 if r == 170 and g == 0 and b == 0:
#                     target_px += 1

#         target_area = target_px / frame_area

#         return target_area


# # register the environment
# register(
#     id="FTO-v1",
#     entry_point=lambda: FTO_Env(),
#     max_episode_steps=1000,
# )

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


# Robot action to go in a certain direction in discrete space
class RobotAction(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    STOP = 3
    TURN_OVER = 4


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

        # Set the action spaces: 0 = left, 1 = right, 2 = forward, 3 = stop
        # case 1: discrete
        # self.action_space: spaces.Discrete = spaces.Discrete(len(RobotAction))

        # case 2: continuous
        self.action_space = spaces.Box(
            low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32
        )

        # Set the observation spaces
        # RGB images of shape (camera_width, camera_height, 3) and a target value in range 0-100
        camera_width = self.camera.getWidth()
        camera_height = self.camera.getHeight()
        self.image_size = camera_width * camera_height * 3  # Flattened image size

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size + 1,),  # Flattened image + 1 for the target value
            dtype=np.float32,
        )

        # environment specification
        self.state = None
        self.camera: Any = None
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
        initial_image = np.zeros(
            (self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8
        ).flatten()
        initial_target_value = np.array([0], dtype=np.float32)
        self.state = np.concatenate([initial_image, initial_target_value])

        # info
        info: dict = {}

        return self.state, info

    def step(self, action: int):
        # perform a continuous action
        self.motors[0].setVelocity(action[0])
        self.motors[1].setVelocity(action[1])

        # Get the new state
        super().step(self.__timestep)

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        target_area = 0  # Initialize target area
        data = np.zeros(
            (height, width, 3), dtype=np.uint8
        ).flatten()  # Initialize image data

        if (
            self.camera.isRecognitionSegmentationEnabled()
            and self.camera.getRecognitionSamplingPeriod() > 0
        ):
            data = self.camera.getRecognitionSegmentationImage()

            if data:
                self.display_segmented_image(data, width, height)

                # calculate the target area
                target_area = self.calculate_target_area(
                    data, width, height, frame_area
                )

        # observation
        data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3)).flatten()
        self.state = np.concatenate([data, np.array([target_area], dtype=np.float32)])

        # check if the episode is done
        done = bool(target_area >= self.target_threshold)

        # reward
        reward = target_area
        if done:
            reward += 1000

        # info
        info = {}

        return self.state.astype(np.int32), reward, done, False, info

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


# register the environment
register(
    id="FTO-v1",
    entry_point=lambda: FTO_Env(),
    max_episode_steps=1000,
)
