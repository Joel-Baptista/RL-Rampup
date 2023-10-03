import gym
from agent_ac import AgentAC
import numpy as np
import torch
import json
from datetime import timedelta
import time
import math

N_EXPERIMENT = 1
INITIAL_EXPERIMENT = 0

def save_data(data, n):

    with open(f"/home/joel/PhD/RL-Skid2Mid/cart_pole/ac/ac{n}_train_data.json", 'w') as outfile:
            json.dump(data, outfile)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    n_games = 500
    i = 0

    train_data = {"train_time": "",
                  "last_epoch": 0,
                  "avg_score": 0}

    n_experiments = N_EXPERIMENT

    for n in range(INITIAL_EXPERIMENT, n_experiments):
        max_avg_score = 0
        agent = AgentAC(4, 2, 0.003)
        scores, eps_history = [], []
        # for i in range(n_games):
        st = time.time()
        j = 0
        is_q_learning = True
        agent.Q_eval.train()
        agent.policy.eval()
        while True:
            i += 1
            score = 0
            done = False
            observation = env.reset()
            
            while not done:
                j += 1
                # frame = env.render(mode='rgb_array')
                
                if j % 500 == 0 and is_q_learning:
                    is_q_learning = False
                    agent.Q_eval.eval()
                    agent.policy.train()
                    agent.saved_log_probs = []
                    agent.Gt = []
                elif j % 500 == 0 and not is_q_learning:
                    is_q_learning = True
                    agent.Q_eval.train()
                    agent.policy.eval()

                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                agent.store_transition(observation, action, reward, observation_, done)

                if is_q_learning: agent.learn_q(observation, action, reward, observation_)
                if not is_q_learning: agent.learn_policy()

                # reward = reward_shaping(observation, action)
                score += reward 

                observation = observation_

            scores.append(score)
            # eps_history.append(agent.epsilion)

            avg_score = np.mean(scores[-300:])
            
            if i % 10 == 0 and avg_score > max_avg_score:
                torch.save(agent.policy.state_dict(), f"/home/joel/PhD/RL-Skid2Mid/cart_pole/ac/model_ac{n}.pt")
                train_data = {"train_time": str(timedelta(seconds=time.time() - st)),
                    "last_epoch": i,
                    "avg_score": avg_score}
                save_data(train_data, n)
                max_avg_score = avg_score

            print('episode: ', i, 'score %.2f' % score, 
                    'average_score %.2f' % avg_score,
                    f"{round((avg_score / max_avg_score) * 100, 2)}",
                    f'max_avg_score: {max_avg_score}',
                    f"epsilon: {agent.epsilon}")
            if avg_score > 450:
                break
        
        train_data = {"train_time": str(timedelta(seconds=time.time() - st)),
                    "last_epoch": i,
                    "avg_score": avg_score}
        save_data(train_data, n)

        x = [i+1 for i in range(n_games)]
        
        torch.save(agent.policy.state_dict(), f"/home/joel/PhD/RL-Skid2Mid/cart_pole/ac/model_ac{n}.pt")
        del agent