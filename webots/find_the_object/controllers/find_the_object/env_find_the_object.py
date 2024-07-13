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

        self.camera = self.getDevice("camera")

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

    def run(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        while self.step(self.__timestep) != -1:
            if (
                self.camera.isRecognitionSegmentationEnabled()
                and self.camera.getRecognitionSamplingPeriod() > 0
            ):
                objects = self.camera.getRecognitionObjects()
                data = self.camera.getRecognitionSegmentationImage()

                if data:
                    self.display_segmented_image(data, width, height)

                    # calculate the target area
                    target_area = self.calculate_target_area(
                        data, width, height, frame_area
                    )

                    done = self.is_done(target_area, 0.35)
                    exit(0) if done else None

                self.objects_recognition(objects, width, target_area)

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

        # set the initial velocity of the motors randomly
        initial_direction = random.choice([-1, 1]) * self.speed / 1.5
        self.left_motor.setVelocity(-initial_direction)
        self.right_motor.setVelocity(initial_direction)

        # set the frame area
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        self.frame_area = self.width * self.height

        # initial state
        self.state = np.zeros(
            (self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8
        )

        # Reset the previous target area
        self.previous_target_area = 0.0

        info = {}

        return self.state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Apply the action to set the velocity of the motors
        left_speed, right_speed = action
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Step the simulation
        super().step(self.__timestep)

        # Get the new observation
        new_obs = self.get_observation()

        if (
            self.camera.isRecognitionSegmentationEnabled()
            and self.camera.getRecognitionSamplingPeriod() > 0
        ):
            objects = self.camera.getRecognitionObjects()
            data = self.camera.getRecognitionSegmentationImage()

            if data:
                self.display_segmented_image(data, self.width, self.height)

                # calculate the target area
                target_area = self.calculate_target_area(
                    data, self.width, self.height, self.frame_area
                )

                # Check if the episode is done
                done = self.is_done(target_area, 0.35)

            self.objects_recognition(objects, self.width, target_area)

        # Calculate the reward
        reward = self.calculate_reward(target_area)

        truncation = False
        info = {}

        # Update the previous target area
        self.previous_target_area = target_area

        # Return the observation, reward, done, truncation, and info
        return new_obs, reward, done, truncation, info

    def calculate_reward(self, target_area: float) -> float:
        # Reward for increasing the target area
        reward = target_area - self.previous_target_area

        # Optional: Add a small penalty for time steps to encourage faster finding
        reward -= 0.01

        return reward

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

    def objects_recognition(self, objects, width, target_area):
        for obj in objects:
            for i in range(obj.getNumberOfColors()):
                r, g, b = obj.getColors()[3 * i : 3 * i + 3]
                print(f"Color {i + 1}/{obj.getNumberOfColors()}: {r} {g} {b}")

                if r == 0.666667 and g == 0 and b == 0:
                    print("Target found, determining position...")
                    print(f"Target area: {target_area*100:.2f}% of the frame")

                    position_on_image = obj.getPositionOnImage()
                    obj_x, obj_y = position_on_image[0], position_on_image[1]

                    print(f"Object position on image: x={obj_x}, y={obj_y}")

                    if obj_x < width / 3:
                        self.turn_left()
                    elif obj_x < 2 * width / 3:
                        self.move_forward()
                    else:
                        self.turn_right()

    def is_done(self, target_area, threshold=0.25):
        done = False
        if target_area >= threshold:
            print(f"Target area meets or exceeds {threshold * 100:.2f}% of the frame.")
            self.stop_motors()
            done = True

        return done

    def stop_motors(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def turn_left(self):
        print("Target on the left, turning left...")
        self.left_motor.setVelocity(-self.speed)
        self.right_motor.setVelocity(self.speed)

    def move_forward(self):
        print("Target in the center, moving forward...")
        self.left_motor.setVelocity(self.speed)
        self.right_motor.setVelocity(self.speed)

    def turn_right(self):
        print("Target on the right, turning right...")
        self.left_motor.setVelocity(self.speed)
        self.right_motor.setVelocity(-self.speed)


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
