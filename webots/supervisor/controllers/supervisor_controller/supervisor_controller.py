from controller import Supervisor # type: ignore

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

bb8_node = robot.getFromDef('BB-8')  # get the BB8 node
translation_field = bb8_node.getField('translation')  # get the translation field of the BB8 node

i = 0
while robot.step(TIME_STEP) != -1:
  if i == 0:
      new_value = [2.5, 0, 0]
      translation_field.setSFVec3f(new_value)

  i += 1