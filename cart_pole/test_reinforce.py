import gym
from agent_reinforce import Agent
import numpy as np
import torch
import json
from datetime import timedelta
import time

N_AGENT = 7
N_GAMES = 10

def save_data(data):

    with open(f'/home/joel/PhD/RL-Skid2Mid/cart_pole/policy_gradients/tests/reinforce{N_AGENT}_test_data.json', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    agent = Agent(4, 2)
    agent.policy.load_state_dict(torch.load(f"/home/joel/PhD/RL-Skid2Mid/cart_pole/policy_gradients/model_reinforce{N_AGENT}.pt"))
    agent.policy.eval()
    scores, eps_history = [], []
    n_games = N_GAMES

    train_data = {"train_time": "",
                  "last_epoch": 0,
                  "avg_score": 0}

    successful = 0
    st = time.time()
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        j = 0
        while not done:
            frame = env.render(mode='rgb_array')

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward 

            observation = observation_

        scores.append(score)
        # eps_history.append(agent.epsilion)

        avg_score = np.mean(scores[-100:])
        
        if score >= 500:
            successful += 1

        print('episode: ', i, 'score %.2f' % score)

    test_data = {"number of episodes": n_games,
                "espisodes completed": successful,
                "average reward": np.mean(scores),
                "max reward": np.max(scores),
                "min reward": np.min(scores),
                "rewards": scores}

    print(test_data)
    # save_data(test_data)
