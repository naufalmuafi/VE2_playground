from controller import Robot, DistanceSensor, Motor

# TIME_STEP is to be used as the simulation time step in milliseconds
TIME_STEP = 64

# create the Robot instance.
robot = Robot()

# initialize devices
ps = []
ps_names = [
    'ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7'
]

for i in range(len(ps_names)):
    ps.append(robot.getDevice(ps_names[i]))
    ps[i].enable(TIME_STEP

# get the devices
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

# set the target position of the motors
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# set up the motor speeds
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    # read sensors outputs
    # process behavior
    # write actuators inputs