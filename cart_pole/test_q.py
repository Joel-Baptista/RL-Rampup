import gym
from agent_buffer import AgentBuffer
import numpy as np
import torch
import json
from datetime import timedelta
import time

N_GAMES = 10
N_AGENT = 7

def save_data(data):

    with open(f'/home/joel/PhD/RL-Skid2Mid/cart_pole/Q_functions/tests/q{N_AGENT}_buffer_test_data.json', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    agent = AgentBuffer(gamma=0.99, epsilon=1.0, batch_size=256, n_actions=2,
        eps_end=0.01, input_dims=[4], lr=0.0001, max_mem_size=100_000)
    agent.Q_eval.load_state_dict(torch.load(f"/home/joel/PhD/RL-Skid2Mid/cart_pole/Q_functions/model_Q_buffer{N_AGENT}.pt"))
    agent.Q_eval.eval()
    scores, eps_history = [], []
    n_games = N_GAMES
 
    

    st = time.time()
    successful = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        j = 0
        while not done:
            frame = env.render(mode='rgb_array')

            action = agent.predict_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward 

            observation = observation_

        if score >= 500:
            successful += 1

        scores.append(score)

        print('episode: ', i, 'score %.2f' % score)

    test_data = {"number of episodes": n_games,
                 "agent number": N_AGENT,
                 "espisodes completed": successful,
                 "average reward": np.mean(scores),
                 "max reward": np.max(scores),
                 "min reward": np.min(scores),
                 "rewards": scores}

    print(test_data)
    save_data(test_data)
    
