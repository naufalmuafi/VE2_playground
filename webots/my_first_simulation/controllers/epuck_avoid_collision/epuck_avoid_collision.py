from controller import Robot, DistanceSensor, Motor

# TIME_STEP is to be used as the simulation time step in milliseconds
TIME_STEP = 64

# create the Robot instance.
robot = Robot()

# initialize devices
# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    # read sensors outputs
    # process behavior
    # write actuators inputs