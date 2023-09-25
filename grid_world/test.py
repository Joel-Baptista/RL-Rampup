from grid_world_env import GridWorld
from grid_agent import GridAgent
from matplotlib import pyplot as plt
from matplotlib import colors
import copy
import numpy as np
import json
from matplotlib.animation import FuncAnimation
import time
global pos
global pos_next

N = 64
AGENT_POS = None

if __name__=="__main__":

    with open(f'history_{N}.json') as f:
        history = json.load(f)
    
    with open(f'agent_{N}.json') as f:
        Q_function = json.load(f)

    episodes = history["episodes"]
    env_props = history["env_props"]
    
    env = GridWorld(grid_size=env_props["grid_size"])
    agent = GridAgent(grid_size=env_props["grid_size"], n_actions=4)
    agent.Q = np.array(Q_function)
    pos, _ = env.reset(goal_pos=tuple(env_props["goal_pos"]), agent_pos=AGENT_POS)

    # states = []
    # done = False
    # while not done:
    #     action = agent.predict(pos)
    #     pos_next, reward, done = env.step(action)
        
    #     states.append(pos_next)
    #     pos = pos_next

    grid_experiment = copy.deepcopy(env.grid)
    grid_experiment[tuple(pos)] = 1

    fig = plt.figure(figsize=(8,8))
    cmap = colors.ListedColormap(['dimgrey','red','limegreen'])
    plt.pcolor(grid_experiment[::-1],cmap=cmap,edgecolors='k', linewidths=3, norm=colors.BoundaryNorm([0, 1, 2], 2))
    

    def update_grid(i):
        global pos
        global pos_next
        
        action = agent.predict(pos)
        pos_next, reward, done = env.step(action)
        
        # agent_pos = states[i]
        grid_experiment = copy.deepcopy(env.grid)
        grid_experiment[tuple(pos_next)] = 1

        pos = pos_next
        plt.cla()
        plt.pcolor(grid_experiment[::-1],cmap=cmap,edgecolors='k', linewidths=3, norm=colors.BoundaryNorm([0, 1, 2], 2))
        fig.canvas.draw()
        

    ani = FuncAnimation(fig, update_grid, interval=200)
    plt.show()   
