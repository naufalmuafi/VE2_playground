# 1: Project


Here is the updated code:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor, Display
import random

class WebotsGymEnv(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()
        random.seed(42)

        # Set the time step
        self.timeStep = 64

        # Set the speed of the motors
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

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: turn left, 1: move forward, 2: turn right
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera.getHeight(), self.camera.getWidth(), 4), dtype=np.uint8)

        self.robot_node = self.getFromDef("ROBOT")
        self.targets = self.getFromDef("TARGET_1")

        self.done = False

    def reset(self):
        self.simulationReset()
        self.step(self.timeStep)
        
        initial_direction = random.choice([-1, 1]) * self.speed / 1.5
        self.left_motor.setVelocity(-initial_direction)
        self.right_motor.setVelocity(initial_direction)
        
        self.done = False
        observation = self.get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if action == 0:
            self.turn_left()
        elif action == 1:
            self.move_forward()
        elif action == 2:
            self.turn_right()

        self.step(self.timeStep)

        observation = self.get_observation()
        target_area = self.calculate_target_area(observation)
        reward = self.compute_reward(target_area)
        terminated = self.is_done(target_area)
        truncated = False  # You can define conditions for truncation if necessary
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_observation(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        data = self.camera.getRecognitionSegmentationImage()
        if data:
            segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
            self.display.imagePaste(segmented_image, 0, 0, False)
            self.display.imageDelete(segmented_image)

        return np.array(data).reshape((height, width, 4))

    def calculate_target_area(self, data):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height
        target_px = 0

        for y in range(height):
            for x in range(width):
                index = (y * width + x) * 4
                b, g, r, a = data[index: index + 4]

                if r == 170 and g == 0 and b == 0:
                    target_px += 1

        return target_px / frame_area

    def compute_reward(self, target_area):
        reward = target_area  # Reward is proportional to the target area detected
        return reward

    def is_done(self, target_area):
        threshold = 0.35
        if target_area >= threshold:
            print(f"Target area meets or exceeds {threshold * 100:.2f}% of the frame.")
            self.stop_motors()
            return True
        return False

    def stop_motors(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def turn_left(self):
        self.left_motor.setVelocity(-self.speed)
        self.right_motor.setVelocity(self.speed)

    def move_forward(self):
        self.left_motor.setVelocity(self.speed)
        self.right_motor.setVelocity(self.speed)

    def turn_right(self):
        self.left_motor.setVelocity(self.speed)
        self.right_motor.setVelocity(-self.speed)


if __name__ == "__main__":
    env = WebotsGymEnv()
    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
```

### Key Updates

1. **Class Definition**:
   - Inherited from both `Supervisor` and `gym.Env`.
   - Called `super().__init__()` to initialize both parent classes.

2. **Method Adjustments**:
   - Removed redundant `self.robot` references.
   - Directly called Webots Supervisor methods like `self.getDevice`.

### Next Steps

**a.** Would you like to test this environment with a reinforcement learning algorithm, such as one from Stable Baselines3?

**b.** Do you need further customization or additional features in the environment?
