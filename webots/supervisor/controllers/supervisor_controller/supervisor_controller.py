from controller import Supervisor # type: ignore

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

bb8_node = robot.getFromDef('BB-8')  # get the BB8 node
translation_field = bb8_node.getField('translation')  # get the translation field of the BB8 node

root_node = robot.getRoot()  # get the root node
children_field = root_node.getField('children')  # get the children field of the root node

# Ball Node
children_field.importMFNodeFromString(-1, 'DEF BALL Ball { translation 0 1 1}')
ball_node = robot.getFromDef('BALL')
color_field = ball_node.getField('color')

i = 0
while robot.step(TIME_STEP) != -1:
  # move the BB-8 robot
  if i == 0:
    new_value = [2.5, 0, 0]
    translation_field.setSFVec3f(new_value)
  
  # removing the robot node
  if i == 10:
    bb8_node.remove()
  
  # spawning the robot
  if i == 20:
    children_field.importMFNodeFromString(-1, 'DEF Nao Nao { translation 2.5 0 0.334 }')
    nao_node = robot.getFromDef('Nao')
  
  ball_position = ball_node.getPosition()
  print(f'Ball position: {ball_position[0]} {ball_position[1]} {ball_position[2]}\n')
  
  if ball_position[2] < 0.2:
    red_color = [1, 0, 0]
    color_field.setSFColor(red_color)
  
  if i == 30:
    children_field.importMFNodeFromString(-1, 'DEF BB-8 BB-8 { translation 2.5 0 0 controller "<none>" }')
    bb8_node = robot.getFromDef('BB-8')
    translation_field = bb8_node.getField('translation')
  
  if i == 40:
    translation_field.setSFVec3f([0, 0, 0])
  
  if i == 50:
    nao_node.remove()
    ball_node.remove()
    robot.worldSave()  # save the world state
    break

  i += 1