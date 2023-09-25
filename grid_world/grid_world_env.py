import numpy as np
import random
import math
import copy

MIN_PLANT_VALUE = -1
MAX_PLANT_VALUE = 0.5
GOAL_VALUE = 10
EDGE_VALUE = -10
VISIBLE_RADIUS = 1

class GridWorld():
    def __init__(self, grid_size=8) -> None:
        self.grid_size = grid_size

    def reset(self, only_agent = False, goal_pos = None, agent_pos = None):

        if not only_agent:
            self.grid = np.zeros((self.grid_size, self.grid_size))
            self.grid_reward = copy.deepcopy(self.grid)

            #Put goal in random corner

            if goal_pos is None:
                corners = [(0, 0), (self.grid_size-1, 0), (0, self.grid_size-1), (self.grid_size-1, self.grid_size-1)]
                self.goal_pos = random.choice(corners)
            else:
                self.goal_pos = goal_pos
            
            self.grid_reward[self.goal_pos] = GOAL_VALUE
            self.grid[self.goal_pos] = 2 

        if agent_pos is None:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            self.agent_pos = (x, y)
        else:
            self.agent_pos = agent_pos

        return self.agent_pos, self.grid_reward[self.agent_pos]

    def step(self, action):

        y, x = self.agent_pos
        if action == 0: y -= 1 # Up
        elif action == 1: x += 1 # Right 
        elif action == 2: y += 1 # Down
        elif action == 3: x -= 1 # Left

        reward = 0
        if x<0: 
            x=0
            reward = EDGE_VALUE
        if x>self.grid_size-1: 
            x=self.grid_size-1
            reward = EDGE_VALUE

        if y<0: 
            y=0
            reward = EDGE_VALUE

        if y>self.grid_size-1: 
            y=self.grid_size-1
            reward = EDGE_VALUE

        self.agent_pos = (y, x)

        done = False
        if self.agent_pos == self.goal_pos:
            done = True

        return self.agent_pos, self.grid_reward[self.agent_pos] + reward, done


