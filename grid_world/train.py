from grid_world_env import GridWorld
from grid_agent import GridAgent
from matplotlib import pyplot as plt
from matplotlib import colors
import copy
import numpy as np
import time

GRID_SIZE = 64
EPSILON_END = 0.2
EPSILON_DECAY = 0.00005
GAMMA = 0.99
EPISODES = 500
CONVERGE_DIST_MULT = 3
SEQUENT_CONVERG = 20

# 0 - UP
# 1 - RIGHT
# 2 - DOWN
# 3 - LEFT

if __name__=="__main__":

    env = GridWorld(grid_size=GRID_SIZE)
    agent = GridAgent(grid_size=GRID_SIZE, n_actions=4,epsilon_end=EPSILON_END,gamma=GAMMA, e_decay=EPSILON_DECAY)
    pos, _ = env.reset()

    # for i in range(1, EPISODES+1):
    i = 0
    count = 0
    st = time.time()
    while True:
        i += 1
        print(f"Episode: {i} - Epsilon: {agent.epsilon}")

        env.reset(only_agent=True)
        done = False
        steps = 0
        while not done:
            steps += 1
            action = agent.choose_action(pos)
            pos_next, reward, done = env.step(action)
            
            agent.q_learn(pos, action, reward, pos_next)

            agent.load_to_history(episode=i, state=pos_next)

            pos = pos_next
        print(f"Agent steps: {steps}")
        if steps < CONVERGE_DIST_MULT * GRID_SIZE: 
            count += 1
            if count > SEQUENT_CONVERG:
                break
        else:
            count = 0

    print(np.max(np.max(agent.Q, axis=2)))
    train_time = time.time() - st
    print(f"Time training: {train_time}")
    agent.save_history(env.goal_pos, GRID_SIZE, train_time)
    agent.save_agent(GRID_SIZE)
