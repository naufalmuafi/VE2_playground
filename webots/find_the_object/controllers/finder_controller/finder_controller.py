"""

finder_controller for finding the object project.

Using Webots and Stable Baselines3.

"""

from controller import Robot, Display


class Controller(Robot):
    def __init__(self):
        # Initialize the Robot class
        super(Controller, self).__init__()

        # Set the time step
        self.timeStep = 64

        # set the speed of the motors
        self.speed = 1.5

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
        self.left_motor.setVelocity(-self.speed)
        self.right_motor.setVelocity(self.speed)

    def run(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()

        while self.step(self.timeStep) != -1:
            if (
                self.camera.isRecognitionSegmentationEnabled()
                and self.camera.getRecognitionSamplingPeriod() > 0
            ):
                objects = self.camera.getRecognitionObjects()
                data = self.camera.getRecognitionSegmentationImage()

                if data:
                    segmented_image = self.display.imageNew(
                        data, Display.BGRA, width, height
                    )
                    self.display.imagePaste(segmented_image, 0, 0, False)
                    self.display.imageDelete(segmented_image)

                for obj in objects:
                    for i in range(obj.getNumberOfColors()):
                        r, g, b = obj.getColors()[3 * i : 3 * i + 3]
                        print(f"Color {i + 1}/{obj.getNumberOfColors()}: {r} {g} {b}")
                        if r == 0.666667 and g == 0 and b == 0:
                            print("Target found")


if __name__ == "__main__":
    controller = Controller()
    controller.run()
