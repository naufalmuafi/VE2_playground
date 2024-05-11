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

class WarehouseRobot:
  # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
  def __init__(self, grid_rows=4, grid_cols=5, fps=1):
    self.grid_rows = grid_rows
    self.grid_cols = grid_cols
    self.reset()
    
    self.fps = fps
    self.last_action = ''
    self._init_pygame()
  
  def _init_pygame(self):
    pygame.init() # initialize pygame
    pygame.display.init() # initialize the display module
    self.clock = pygame.time.Clock() # create a clock object to control the frame rate
    
    # default font
    self.action_font = pygame.font.SysFont('Calibre', 30)
    self.action_info_height = self.action_font.get_height()
    
    # for rendering the grid purposes
    self.cell_height = 64
    self.cell_width = 64
    self.cell_size = (self.cell_width, self.cell_height)
    
    # define game window size (w, h)
    self.window_size = (self.grid_cols * self.cell_width, self.cell_height * self.grid_rows + self.action_info_height)
    
    # initialize game windows
    self.window_surface = pygame.display.set_mode(self.window_size)
    
    # load and resize sprites
    file_name = path.join(path.dirname(__file__), 'sprites/bot_blue.png')
    img = pygame.image.load(file_name)
    self.robot_img = pygame.transform.scale(img, self.cell_size)
    
    file_name = path.join(path.dirname(__file__), 'sprites/floor.png')
    img = pygame.image.load(file_name)
    self.floor_img = pygame.transform.scale(img, self.cell_size)
    
    file_name = path.join(path.dirname(__file__), 'sprites/package.png')
    img = pygame.image.load(file_name)
    self.goal_img = pygame.transform.scale(img, self.cell_size)
  
  def reset (self, seed=None):
    # initialize robot's starting position
    self.robot_pos = [0, 0]
    
    # random target position
    random.seed(seed)
    self.target_pos = [
      random.randint(1, self.grid_rows - 1),
      random.randint(1, self.grid_cols - 1)
    ]
  
  def action(self, robot_action:RobotAction) -> bool:
    self.last_action = robot_action
    
    # move the robot to the next cell
    if robot_action == RobotAction.LEFT:
      if self.robot_pos[1] > 0:
        self.robot_pos[1] -= 1
    elif robot_action == RobotAction.RIGHT:
      if self.robot_pos[1] < self.grid_cols - 1:
        self.robot_pos[1] += 1
    elif robot_action == RobotAction.UP:
      if self.robot_pos[0] > 0:
        self.robot_pos[0] -= 1
    elif robot_action == RobotAction.DOWN:
      if self.robot_pos[0] < self.grid_rows - 1:
        self.robot_pos[0] += 1
        
    # return True if Robot reaches the Target
    return self.robot_pos == self.target_pos