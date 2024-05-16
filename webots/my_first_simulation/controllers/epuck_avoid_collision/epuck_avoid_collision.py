from controller import Robot, DistanceSensor, Motor

# TIME_STEP is to be used as the simulation time step in milliseconds
TIME_STEP = 64

# MAX_SPEED is to be used as the maximum speed of the motors in rad/s
MAX_SPEED = 6.28

# create the Robot instance.
robot = Robot()

# initialize devices
ps = []
ps_names = [
    'ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7'
]

# get the sensors
for i in range(len(ps_names)):
    ps.append(robot.getDevice(ps_names[i]))
    ps[i].enable(TIME_STEP)

# get the motors
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
    ps_values = []
    for i in range(len(ps)):
        ps_values.append(ps[i].getValue())
    
    # detect obstacles
    right_obstacle = ps_values[0] > 80.0 or ps_values[1] > 80.0 or ps_values[2] > 80.0
    left_obstacle = ps_values[5] > 80.0 or ps_values[6] > 80.0 or ps_values[7] > 80.0

    # process behavior    
    # initialize motor speeds at 50% of MAX_SPEED
    left_speed = 0.5 * MAX_SPEED
    right_speed = 0.5 * MAX_SPEED

    # modify speeds according to obstacles
    if left_obstacle:
        # turn right
        left_speed = 0.5 * MAX_SPEED
        right_speed = -0.5 * MAX_SPEED
    elif right_obstacle:
        # turn left
        left_speed = -0.5 * MAX_SPEED
        right_speed = 0.5 * MAX_SPEED
    
    # write actuators inputs
    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)