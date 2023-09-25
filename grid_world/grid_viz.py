import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import json
import copy
import time
import math

TIME = 3
TRAIN_RATIO = 1
N = 16

with open(f'history_{N}.json') as f:
	history = json.load(f)

episodes = history["episodes"]
env_props = history["env_props"]
grid = np.zeros((env_props["grid_size"], env_props["grid_size"]))
grid[tuple(env_props["goal_pos"])] = 2
cmap = colors.ListedColormap(['dimgrey','red','limegreen'])
agent_pos = episodes["1"][0]
grid_experiment = copy.deepcopy(grid)
grid_experiment[tuple(agent_pos)] = 1

fig = plt.figure(figsize=(8,8))

total_eps = len(list(episodes.keys()))

episode =  math.ceil(TRAIN_RATIO * total_eps)
if episode == 0: episode = 1
steps = len(episodes[str(episode)])
update_time = (TIME / steps) * 1000

# if update_time < 50: update_time = 50

def update_grid(i):
    agent_pos = episodes[str(episode)][i]
    grid_experiment = copy.deepcopy(grid)
    grid_experiment[tuple(agent_pos)] = 1

    plt.cla()
    plt.title(f"Episode: {episode}, N steps: {steps}")
    plt.pcolor(grid_experiment[::-1],cmap=cmap,edgecolors='k', linewidths=3, norm=colors.BoundaryNorm([0, 1, 2], 2))
    fig.canvas.draw()

ani = FuncAnimation(fig, update_grid, interval=update_time)
plt.show()   
