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

        # threshold for the target area
        self.frame_threshold = 0.35

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

    def run(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        red = 0
        green = 0
        blue = 0

        # Create a 2D list to store pixel values
        pixels = []

        while self.step(self.timeStep) != -1:
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

                    # Check if the target area meets the threshold
                    done = self.is_done(target_area, self.frame_threshold)
                    print(data.shape) if done else None
                    exit(0) if done else None

                self.objects_recognition(objects, width, target_area)

                # Display the pixel values
                # for row in pixels:
                #     print(row)

    def display_segmented_image(self, data, width, height):
        segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
        self.display.imagePaste(segmented_image, 0, 0, False)
        self.display.imageDelete(segmented_image)

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
