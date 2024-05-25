from controller import Supervisor # type: ignore

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

# [CODE PLACEHOLDER 1]

i = 0
while robot.step(TIME_STEP) != -1:
  # [CODE PLACEHOLDER 2]

  i += 1