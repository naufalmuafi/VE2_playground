'''
This module models the problem to be solved. In this very simple example, the problem is to optimze a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.
'''

import sys
import pygame
import random
from os import path
from enum import Enum

# actions the Robot is capable of performing i.e. go in a certain direction
class RobotAction(Enum):
  LEFT = 0
  DOWN = 1
  RIGHT = 2
  UP = 3

# The Warehouse is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
  FLOOR = 0
  ROBOT = 1
  TARGET = 2
  
  # return the first letter of tile name, for printing to the console
  def __str__(self):
    return self.name[:1]

