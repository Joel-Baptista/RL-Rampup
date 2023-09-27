import gym
from agent import Agent
from agent_buffer import AgentBuffer
import numpy as np
import torch
from datetime import timedelta
import json
import time

def save_data(data, number = ""):

    with open(f'/home/joel/PhD/RL-Skid2Mid/cart_pole/Q_functions/q{number}_buffer_train_data.json', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # agent = Agent(dim_states=4, n_actions=2, lr=0.0001, gamma=0.99, epsilon_end=0.1, e_decay=0.00005)
    
   
    n_games = 500
    n_experiments = 10

    for n in range(0, n_experiments):
        agent = AgentBuffer(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2,
            eps_end=0.02, input_dims=[4], lr=0.0001, max_mem_size=100_000)
        scores, eps_history = [], []
        st = time.time()
        i = 0
        max_avg_score = 0
        # for i in range(n_games):
        while True:
            score = 0
            done = False
            observation = env.reset()
            i += 1
            while not done:
                # frame = env.render(mode='rgb_array')

                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)

                # reward = reward_shaping(observation, action)
                score += reward 

                agent.store_transition(observation, action, reward, observation_, done)

                agent.learn(observation, action, reward, observation_)

                observation = observation_


            scores.append(score)
            # eps_history.append(agent.epsilion)

            avg_score = np.mean(scores[-100:])
        
            if i % 10 == 0 and avg_score > max_avg_score:
                torch.save(agent.Q_eval.state_dict(), f"/home/joel/PhD/RL-Skid2Mid/cart_pole/Q_functions/model_Q_buffer{n}.pt")
                train_data = {"train_time": str(timedelta(seconds=time.time() - st)),
                    "last_epoch": i,
                    "avg_score": avg_score}
                save_data(train_data, str(n))
                max_avg_score = avg_score

            print('episode: ', i, 'score %.2f' % score, 
                    'average_score %.2f' % avg_score,
                    'epsilon %.2f' % agent.epsilon,
                    f"{round((avg_score / max_avg_score) * 100, 2)}",
                    f'max_avg_score: {max_avg_score}')

            if avg_score > 450 or (avg_score < 0.95 * max_avg_score and avg_score > 100):
                break

        train_data = {"train_time": str(timedelta(seconds=time.time() - st)),
                    "last_epoch": i,
                    "avg_score": avg_score}
        save_data(train_data, str(n))


        x = [i+1 for i in range(n_games)]
        
        torch.save(agent.Q_eval.state_dict(), f"/home/joel/PhD/RL-Skid2Mid/cart_pole/Q_functions/model_Q_buffer{n}.pt")
        del agent
