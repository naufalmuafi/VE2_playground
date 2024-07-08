import cv2
import numpy as np
from controller import Robot, Camera, Display, Keyboard

NB_FILTERS = 6

RED = 0
GREEN = 1
BLUE = 2
YELLOW = 3
PURPLE = 4
WHITE = 5
NONE = 6
ALL = 7

# The scalars correspond to HSV margin (In the first example, [0,5] is the accepted hue for the red filter,
# [150,255] the accepted saturation and [30,255] the accepted value).
lMargin = [
    (0, 150, 30),
    (58, 150, 30),
    (115, 150, 30),
    (28, 150, 30),
    (148, 150, 30),
    (0, 0, 50),
]
uMargin = [
    (5, 255, 255),
    (62, 255, 255),
    (120, 255, 255),
    (32, 255, 255),
    (152, 255, 255),
    (0, 0, 255),
]


def display_commands():
    print("Press R to apply/remove a red filter.")
    print("Press G to apply/remove a green filter.")
    print("Press B to apply/remove a blue filter.")
    print("Press Y to apply/remove a yellow filter.")
    print("Press P to apply/remove a purple filter.")
    print("Press W to apply/remove a white filter.")
    print("Press A to apply all filters.")
    print("Press X to remove all filters.")
    print(
        "When one or several filter is applied, only the corresponding colors are considered in the image."
    )
    print("The processed image consists of the entire image if no filter is used.")


def process_image(image, width, height, filters):
    img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create an empty filtered image
    filtered = np.zeros_like(img)
    filtered[:, :, 3] = 255  # Set alpha channel to 255

    if filters[NB_FILTERS]:
        filtered = img
    else:
        for f in range(NB_FILTERS):
            if filters[f]:
                temp_filtered = cv2.inRange(hsv, lMargin[f], uMargin[f])
                filtered[temp_filtered == 255] = img[temp_filtered == 255]

    return filtered.tobytes()


def apply_filter(filter, filters):
    if filter > NB_FILTERS + 1:
        print("Error: Unknown filter.")
    else:
        cnt = 0

        if filter == NONE:
            filters = [False] * NB_FILTERS
        elif filter == ALL:
            filters = [True] * NB_FILTERS
        else:
            filters[filter] = not filters[filter]

        print("Filters currently applied: ", end="")
        for i in range(NB_FILTERS):
            if filters[i]:
                color_name = ["red", "green", "blue", "yellow", "purple", "white"][i]
                print(color_name, end=" ")
                cnt += 1

        if not cnt:
            print("none (the entire image will be displayed).", end="")
            filters[NB_FILTERS] = True
        else:
            filters[NB_FILTERS] = False

        print()
    return filters


def main():
    # Initialize Webots
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    print("Vision module demo, using openCV.")

    display_commands()

    # Initialize camera
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    width = camera.getWidth()
    height = camera.getHeight()

    # Initialize display
    processed_image_display = robot.getDevice("proc_im_display")
    processed_image = bytearray(4 * width * height)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    filters = [False] * (NB_FILTERS + 1)
    filters[NB_FILTERS] = True

    while robot.step(timestep) != -1:
        key = keyboard.getKey()
        if key >= 0:
            if key == ord("X"):
                filters = apply_filter(NONE, filters)
            elif key == ord("R"):
                filters = apply_filter(RED, filters)
            elif key == ord("G"):
                filters = apply_filter(GREEN, filters)
            elif key == ord("B"):
                filters = apply_filter(BLUE, filters)
            elif key == ord("Y"):
                filters = apply_filter(YELLOW, filters)
            elif key == ord("P"):
                filters = apply_filter(PURPLE, filters)
            elif key == ord("W"):
                filters = apply_filter(WHITE, filters)
            elif key == ord("A"):
                filters = apply_filter(ALL, filters)

        processed_image = process_image(camera.getImage(), width, height, filters)

        # Display the image
        processed_image_ref = processed_image_display.imageNew(
            width, height, processed_image, Display.ARGB
        )
        processed_image_display.imagePaste(processed_image_ref, 0, 0, False)


if __name__ == "__main__":
    main()
