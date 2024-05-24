from controller import Robot, DistanceSensor, Motor # type: ignore

# TIME_STEP is to be used as the simulation time step in milliseconds
TIME_STEP = 64

# MAX_SPEED is to be used as the maximum speed of the motors in rad/s
MAX_SPEED = 1.00

# create the Robot instance.
robot = Robot()

# initialize distance sensors
ds = []
ds_names = ['ds_right', 'ds_left']
for i in range(len(ds_names)):
  ds.append(robot.getDevice(ds_names[i]))
  ds[i].enable(TIME_STEP)

# initialize motors/wheels
wheels = []
wheels_names = ['wheel_1', 'wheel_2', 'wheel_3', 'wheel_4']
for i in range(len(wheels_names)):
  # get the motors devices
  wheels.append(robot.getDevice(wheels_names[i]))
  # set the target position of the motors
  wheels[i].setPosition(float('inf'))
  # set up the motor speeds
  wheels[i].setVelocity(0.0)

# initialize avoid obstacle counter
avoid_obstacle_counter = 0

# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
  # set up the initial direction speeds
  left_speed = MAX_SPEED
  right_speed = MAX_SPEED
  
  if avoid_obstacle_counter > 0:
    avoid_obstacle_counter -= 1
    left_speed = MAX_SPEED
    right_speed = -MAX_SPEED
  else:
    # read sensors outputs
    for i in range(len(ds)):
      # print(f"Sensor {i} value: {ds[i].getValue()}")
      if ds[i].getValue() < 950.0:
        avoid_obstacle_counter = 100
  
  # set the velocity of the motors
  wheels[0].setVelocity(left_speed)
  wheels[1].setVelocity(right_speed)
  wheels[2].setVelocity(left_speed)
  wheels[3].setVelocity(right_speed)        