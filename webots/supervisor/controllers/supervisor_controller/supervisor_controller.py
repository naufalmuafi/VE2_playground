from controller import Supervisor # type: ignore

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

bb8_node = robot.getFromDef('BB-8')  # get the BB8 node
translation_field = bb8_node.getField('translation')  # get the translation field of the BB8 node

root_node = robot.getRoot()  # get the root node
children_field = root_node.getField('children')  # get the children field of the root node

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
    children_field.importMFNodeFromString(-1, 'Nao { translation 2.5 0 0.334 }')

  i += 1