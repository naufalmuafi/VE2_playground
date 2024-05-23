from controller import Robot, DistanceSensor, Motor # type: ignore

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

    # print sensor values
    print(f"Sensor values: {ps_values}")
    
    # detect obstacles
    r = 80.0 # lower the value to detect obstacles further away
    right_obstacle = ps_values[0] > r or ps_values[1] > r or ps_values[2] > r
    left_obstacle = ps_values[5] > r or ps_values[6] > r or ps_values[7] > r

    # process behavior    
    # initialize motor speeds at n% of MAX_SPEED
    n = 0.5    
    left_speed = n * MAX_SPEED
    right_speed = n * MAX_SPEED

    # modify speeds according to obstacles
    if left_obstacle:
        # turn right
        left_speed = n * MAX_SPEED
        right_speed = -n * MAX_SPEED
        print("turn right\n")
    elif right_obstacle:
        # turn left
        left_speed = -n * MAX_SPEED
        right_speed = n * MAX_SPEED
        print("turn left\n")
    
    # write actuators inputs
    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)