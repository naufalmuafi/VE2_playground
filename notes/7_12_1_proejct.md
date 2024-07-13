# 1: The Requirement

Question:


okay, this is my code:

finder_controller.py

```python
"""

finder_controller for finding the object project.

Using Webots and Stable Baselines3.

"""

import random
from controller import Supervisor, Display


class Controller(Supervisor):
    def __init__(self):
        # Initialize the Robot class
        super(Controller, self).__init__()
        random.seed(42)

        # Set the time step
        self.timeStep = 64

        # set the speed of the motors
        self.speed = 1.3

        # Get the camera device
        self.camera = self.getDevice("camera")
        self.camera.enable(self.timeStep)
        self.camera.recognitionEnable(self.timeStep)
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

        self.robot_node = self.getFromDef("ROBOT")
        self.targets = self.getFromDef("TARGET_1")

    def run(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        while self.step(self.timeStep) != -1:
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


if __name__ == "__main__":
    controller = Controller()
    controller.run()
```

I've create a main.py
main.py

```python
import os
import env_find_the_object
import gymnasium as gym
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


TIMESTEPS = 1000

# where to store trained model and logs
model_dir_name = "models_fto"
log_dir_name = "logs_fto"

# define the environment
env = gym.make("FTO-v1")


def create_dir(model_name: str = "models", log_name: str = "logs") -> Tuple[str, str]:
    # create the folder from path
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(log_name, exist_ok=True)

    return model_name, log_name


def check_environment(env: gym.Env) -> None:
    # check the environment
    try:
        print(f"Check the environment: {env}...")
        check_env(env)
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Please check the environment and try again.")
        exit(1)


def train_PPO(env: gym.Env, model_dir: str, log_dir: str, timesteps: int) -> None:
    # use Proximal Policy Optimization (PPO) algorithm
    # use MLP policy for observation space 1D-vector
    print("Training the model with PPO...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    # train and save the model
    model.learn(total_timesteps=timesteps)
    model.save(f"{model_dir}/ppo_{timesteps}")


def test_PPO(env: gym.Env, model_dir: str, timesteps: int = TIMESTEPS) -> None:
    # load the model
    try:
        model = PPO.load(f"{model_dir}/ppo_{timesteps}", env=env)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return

    print("Load Model Successful") if model else None

    # run a test
    obs, _ = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)

        if done:
            obs, _ = env.reset()
            print("Test Terminated.")

        print("Test Successful") if i == timesteps else None


# helper function to wait for user input
def wait_for_y() -> None:
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # create directories
    model_dir, log_dir = create_dir(model_dir_name, log_dir_name)

    # check the environment
    check_environment(env)
    print(f"Environment is ready: {env}")

    # train and test the model with A2C algorithm
    train_PPO(env, model_dir, log_dir, TIMESTEPS)

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("Test the Environment with Predicted Value")

    test_PPO(env, model_dir, TIMESTEPS)
```

environment:

- this is the webots world that contains robot, 5 green object, and 1 red target

objective:

- first, the robot turn to the left/right to search a target with:

```python
# set the initial velocity of the motors randomly
        initial_direction = random.choice([-1, 1]) * self.speed / 1.5
        self.left_motor.setVelocity(-initial_direction)
        self.right_motor.setVelocity(initial_direction)
```

- then, after target is founds, the robot moving towards it, but have a rule
- 1) if target position is in the left side of the frame, so the robot turn left
- 2) if target position is in the center of the frame, so the robot move forward
- 3) if target position is in the right side of the frame, so the robot turn right
- 4) if target_area in frame is meets the threshold (in code is 35%), so it's done

action space:

- the action is set the velocity of the robot based on the rules above

observation space:

- the observation is camera/vision frame from the robot so maybe (width, height, 3) with the value is 0 to 255

reward function:

- honestly, I'm still confuse about this, can you help me?
- but, basically agent rewarded when he's found the target and moving towards it based on the rules. then, get penalized if the agent not found the target and move into the wrong side




Answer:


Let's convert your code into a reinforcement learning setup using Stable-Baselines3, Gymnasium, and Webots. Below are the steps we need to follow:

1. **Create a Custom Gymnasium Environment**: This involves defining the observation space, action space, reward function, and episode termination conditions.
2. **Integrate the Environment with Stable-Baselines3**: Train and test the model using PPO.
3. **Update Your Existing Controller Code**: Modify the `Controller` class to interact with the Gymnasium environment.

### Step 1: Create a Custom Gymnasium Environment

First, create a new file called `env_find_the_object.py`:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from finder_controller import Controller

class FindObjectEnv(gym.Env):
    def __init__(self):
        super(FindObjectEnv, self).__init__()
        self.controller = Controller()
        self.timeStep = self.controller.timeStep
        self.camera = self.controller.camera

        # Define the observation space (image from the camera)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8)

        # Define the action space (0: turn left, 1: move forward, 2: turn right)
        self.action_space = spaces.Discrete(3)

        # Internal variables
        self.done = False

    def reset(self):
        self.controller.simulationReset()
        self.done = False
        self.controller.run()  # Initialize the controller

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        if action == 0:
            self.controller.turn_left()
        elif action == 1:
            self.controller.move_forward()
        elif action == 2:
            self.controller.turn_right()

        self.controller.step(self.timeStep)

        obs = self._get_observation()
        reward = self._calculate_reward()
        self.done = self._is_done()

        return obs, reward, self.done, False, {}

    def _get_observation(self):
        image = self.camera.getImage()
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        data = np.frombuffer(image, np.uint8).reshape((height, width, 4))[:, :, :3]  # Extract only RGB channels
        return data

    def _calculate_reward(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        data = self.camera.getRecognitionSegmentationImage()

        if data:
            target_area = self.controller.calculate_target_area(data, width, height, frame_area)
            if target_area >= 0.35:
                return 100  # Reward for finding the target
            else:
                return 1  # Small reward for getting closer to the target

        return -1  # Penalty for not finding the target

    def _is_done(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        data = self.camera.getRecognitionSegmentationImage()

        if data:
            target_area = self.controller.calculate_target_area(data, width, height, frame_area)
            if target_area >= 0.35:
                return True

        return False
```

### Step 2: Integrate the Environment with Stable-Baselines3

Update your `main.py` to include this new environment and ensure it works with Stable-Baselines3:

```python
import os
import env_find_the_object
import gymnasium as gym
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

TIMESTEPS = 10000  # Increased for better training

# where to store trained model and logs
model_dir_name = "models_fto"
log_dir_name = "logs_fto"

# define the environment
env = env_find_the_object.FindObjectEnv()


def create_dir(model_name: str = "models", log_name: str = "logs") -> Tuple[str, str]:
    # create the folder from path
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(log_name, exist_ok=True)

    return model_name, log_name


def check_environment(env: gym.Env) -> None:
    # check the environment
    try:
        print(f"Check the environment: {env}...")
        check_env(env)
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Please check the environment and try again.")
        exit(1)


def train_PPO(env: gym.Env, model_dir: str, log_dir: str, timesteps: int) -> None:
    # use Proximal Policy Optimization (PPO) algorithm
    # use CNN policy for observation space (image)
    print("Training the model with PPO...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    # train and save the model
    model.learn(total_timesteps=timesteps)
    model.save(f"{model_dir}/ppo_{timesteps}")


def test_PPO(env: gym.Env, model_dir: str, timesteps: int = TIMESTEPS) -> None:
    # load the model
    try:
        model = PPO.load(f"{model_dir}/ppo_{timesteps}", env=env)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return

    print("Load Model Successful") if model else None

    # run a test
    obs, _ = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)

        if done:
            obs, _ = env.reset()
            print("Test Terminated.")
            break

        print("Test Successful") if i == timesteps - 1 else None


# helper function to wait for user input
def wait_for_y() -> None:
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # create directories
    model_dir, log_dir = create_dir(model_dir_name, log_dir_name)

    # check the environment
    check_environment(env)
    print(f"Environment is ready: {env}")

    # train and test the model with PPO algorithm
    train_PPO(env, model_dir, log_dir, TIMESTEPS)

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("Test the Environment with Predicted Value")

    test_PPO(env, model_dir, TIMESTEPS)
```

### Step 3: Update Your Existing Controller Code

Finally, make sure your `finder_controller.py` is adapted for use within the Gym environment. The main changes here are to ensure the `Controller` class methods are ready to be used by the Gym environment.

The `Controller` class is largely unchanged, but ensure you have proper initialization and handling methods for interaction with the custom Gym environment.

### Summary

The environment `FindObjectEnv` manages the interaction with the robot through Webots, handles observations from the camera, decides actions (left, forward, right), and calculates rewards. The main training script in `main.py` sets up the environment, checks it, trains the PPO model, and tests it.

Please make sure that your Webots setup is properly configured to support the `finder_controller` and other components, as the custom environment relies heavily on the robot simulation running smoothly.

Let me know if you encounter any issues or need further adjustments!


# 2: how about if the action is continuous?


If the action space is continuous, instead of using discrete actions (left, forward, right), the robot can take any velocity value within a specified range for both motors. This is more flexible and allows the reinforcement learning algorithm to find optimal motor velocities for navigation.

To use continuous actions, you need to change the action space from `Discrete` to `Box` in your custom Gym environment. Here's how to do it:

### Step 1: Modify the Custom Gym Environment

In `env_find_the_object.py`, change the action space and update the `step` method to handle continuous actions:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from finder_controller import Controller

class FindObjectEnv(gym.Env):
    def __init__(self):
        super(FindObjectEnv, self).__init__()
        self.controller = Controller()
        self.timeStep = self.controller.timeStep
        self.camera = self.controller.camera

        # Define the observation space (image from the camera)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera.getHeight(), self.camera.getWidth(), 3), dtype=np.uint8)

        # Define the action space (continuous: left motor speed, right motor speed)
        self.action_space = spaces.Box(low=-self.controller.speed, high=self.controller.speed, shape=(2,), dtype=np.float32)

        # Internal variables
        self.done = False

    def reset(self):
        self.controller.simulationReset()
        self.done = False
        self.controller.run()  # Initialize the controller

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        left_speed, right_speed = action

        self.controller.left_motor.setVelocity(left_speed)
        self.controller.right_motor.setVelocity(right_speed)

        self.controller.step(self.timeStep)

        obs = self._get_observation()
        reward = self._calculate_reward()
        self.done = self._is_done()

        return obs, reward, self.done, False, {}

    def _get_observation(self):
        image = self.camera.getImage()
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        data = np.frombuffer(image, np.uint8).reshape((height, width, 4))[:, :, :3]  # Extract only RGB channels
        return data

    def _calculate_reward(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        data = self.camera.getRecognitionSegmentationImage()

        if data:
            target_area = self.controller.calculate_target_area(data, width, height, frame_area)
            if target_area >= 0.35:
                return 100  # Reward for finding the target
            else:
                return 1  # Small reward for getting closer to the target

        return -1  # Penalty for not finding the target

    def _is_done(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        data = self.camera.getRecognitionSegmentationImage()

        if data:
            target_area = self.controller.calculate_target_area(data, width, height, frame_area)
            if target_area >= 0.35:
                return True

        return False
```

### Step 2: Update the Training Script

In `main.py`, make sure the environment is set up correctly to use the continuous action space:

```python
import os
import env_find_the_object
import gymnasium as gym
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

TIMESTEPS = 10000  # Increased for better training

# where to store trained model and logs
model_dir_name = "models_fto"
log_dir_name = "logs_fto"

# define the environment
env = env_find_the_object.FindObjectEnv()


def create_dir(model_name: str = "models", log_name: str = "logs") -> Tuple[str, str]:
    # create the folder from path
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(log_name, exist_ok=True)

    return model_name, log_name


def check_environment(env: gym.Env) -> None:
    # check the environment
    try:
        print(f"Check the environment: {env}...")
        check_env(env)
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Please check the environment and try again.")
        exit(1)


def train_PPO(env: gym.Env, model_dir: str, log_dir: str, timesteps: int) -> None:
    # use Proximal Policy Optimization (PPO) algorithm
    # use MLP policy for observation space 1D-vector
    print("Training the model with PPO...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    # train and save the model
    model.learn(total_timesteps=timesteps)
    model.save(f"{model_dir}/ppo_{timesteps}")


def test_PPO(env: gym.Env, model_dir: str, timesteps: int = TIMESTEPS) -> None:
    # load the model
    try:
        model = PPO.load(f"{model_dir}/ppo_{timesteps}", env=env)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return

    print("Load Model Successful") if model else None

    # run a test
    obs, _ = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)

        if done:
            obs, _ = env.reset()
            print("Test Terminated.")
            break

        print("Test Successful") if i == timesteps - 1 else None


# helper function to wait for user input
def wait_for_y() -> None:
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # create directories
    model_dir, log_dir = create_dir(model_dir_name, log_dir_name)

    # check the environment
    check_environment(env)
    print(f"Environment is ready: {env}")

    # train and test the model with PPO algorithm
    train_PPO(env, model_dir, log_dir, TIMESTEPS)

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("Test the Environment with Predicted Value")

    test_PPO(env, model_dir, TIMESTEPS)
```

### Step 3: Update Your Existing Controller Code

Ensure the `Controller` class in `finder_controller.py` can handle continuous velocity settings. The main methods are already set to use continuous velocities, so there shouldn't be much to change.

### Summary

By using continuous actions, the robot can adjust its motor speeds more flexibly, which can potentially lead to better performance. This setup allows the PPO algorithm to find optimal motor velocities for navigation towards the target.

Make sure to adjust the reward function and termination conditions as necessary to fit your specific goals. If you need further assistance, feel free to ask!


# 4


Got it! Let's update the `env_find_the_object.py` with the provided `__init__` method and ensure the rest of the class is correctly adjusted to interact with the `Controller`-like setup.

### Updated `env_find_the_object.py`

Here's the updated code for `env_find_the_object.py` with the provided initialization and necessary adjustments:

```python
import gymnasium as gym
from gymnasium import spaces, EnvSpec
import numpy as np
from controller import Supervisor


class FindObjectEnv(Supervisor, gym.Env):
    def __init__(self):
        # Initialize the Robot class
        super().__init__()

        # Register the Environment
        self.spec: EnvSpec = EnvSpec(id="FTO-v1")

        # Set the time step
        self.__timestep: int = int(self.getBasicTimeStep())

        # Initialize the Robot class
        random.seed(42)

        # Set the speed of the motors
        self.speed = 1.3

        # Get the camera devices
        self.camera = self.getDevice("camera")
        self.camera.enable(self.__timestep)
        self.camera.recognitionEnable(self.__timestep)
        self.camera.enableRecognitionSegmentation()

        # Set the threshold of the target area
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

        # Get the display device
        self.display = self.getDevice("segmented image display")

        # Get the left and right wheel motors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        # Set the initial velocity of the motors randomly
        initial_direction = random.choice([-1, 1]) * self.speed / 1.5
        self.left_motor.setVelocity(-initial_direction)
        self.right_motor.setVelocity(initial_direction)

        # Initialize robot and target positions
        self.robot_node = self.getFromDef("ROBOT")
        self.targets = self.getFromDef("TARGET_1")
        self.max_distance = np.sqrt(1.5**2 + 1.5**2)  # Assuming a 1.5x1.5 arena

        # Initialize the target area
        self.previous_target_area = 0.0

    def reset(self):
        self.simulationReset()
        self.step(self.__timestep)

        # Get initial observation
        obs = self._get_observation()
        self.previous_target_area = 0.0
        return obs, {}

    def step(self, action):
        left_speed, right_speed = action

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        self.step(self.__timestep)

        # Get the new observation
        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self._is_done()

        return obs, reward, done, False, {}

    def _get_observation(self):
        image = self.camera.getImage()
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        data = np.frombuffer(image, np.uint8).reshape((height, width, 4))[:, :, :3]  # Extract only RGB channels
        return data

    def _calculate_reward(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        data = self.camera.getRecognitionSegmentationImage()

        if data:
            target_area = self.calculate_target_area(data, width, height, frame_area)
            reward = (target_area - self.previous_target_area) * 1000  # Scaled reward based on area change
            self.previous_target_area = target_area

            if target_area >= self.threshold:
                reward += 100  # Additional reward for meeting the threshold
            return reward

        return -1  # Penalty for not finding the target

    def _is_done(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        data = self.camera.getRecognitionSegmentationImage()

        if data:
            target_area = self.calculate_target_area(data, width, height, frame_area)
            if target_area >= self.threshold:
                return True

        return False

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
```

### Updated `main.py`

Ensure the training and testing scripts in `main.py` interact correctly with the environment.

```python
import os
import env_find_the_object
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

TIMESTEPS = 10000  # Increased for better training

# where to store trained model and logs
model_dir_name = "models_fto"
log_dir_name = "logs_fto"

# define the environment
env = env_find_the_object.FindObjectEnv()


def create_dir(model_name: str = "models", log_name: str = "logs") -> Tuple[str, str]:
    # create the folder from path
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(log_name, exist_ok=True)

    return model_name, log_name


def check_environment(env: env_find_the_object.FindObjectEnv) -> None:
    # check the environment
    try:
        print(f"Check the environment: {env}...")
        check_env(env)
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Please check the environment and try again.")
        exit(1)


def train_PPO(env: env_find_the_object.FindObjectEnv, model_dir: str, log_dir: str, timesteps: int) -> None:
    # use Proximal Policy Optimization (PPO) algorithm
    # use CNN policy for observation space (image)
    print("Training the model with PPO...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    # train and save the model
    model.learn(total_timesteps=timesteps)
    model.save(f"{model_dir}/ppo_{timesteps}")


def test_PPO(env: env_find_the_object.FindObjectEnv, model_dir: str, timesteps: int = TIMESTEPS) -> None:
    # load the model
    try:
        model = PPO.load(f"{model_dir}/ppo_{timesteps}", env=env)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return

    print("Load Model Successful") if model else None

    # run a test
    obs, _ = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)

        if done:
            obs, _ = env.reset()
            print("Test Terminated.")
            break

        print("Test Successful") if i == timesteps - 1 else None


# helper function to wait for user input
def wait_for_y() -> None:
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # create directories
    model_dir, log_dir = create_dir(model_dir_name, log_dir_name)

    # check the environment
    check_environment(env)
    print(f"Environment is ready: {env}")

    # train and test the model with PPO algorithm
    train_PPO(env, model_dir, log_dir, TIMESTEPS)

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("Test the Environment with Predicted Value")

    test_PPO(env, model_dir, TIMESTEPS)
```

### Summary

1. **`env_find_the_object.py`**: Converted the functionality of the `run` method into a Gym environment, including observation space, action space, reward calculation, and done condition, using the provided initialization.
2. **`main.py`**: Ensured that the training and testing scripts interact correctly with the environment, creating directories, checking the environment, training with PPO, and testing the trained model.

This setup should now allow you to train and test your RL agent using the converted Gym environment. Let me know if you have any further questions or need more adjustments!
