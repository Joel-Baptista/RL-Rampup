import gym
from agent_reinforce import Agent
import numpy as np
import torch
import json
from datetime import timedelta
import time
import math
import wandb

N_EXPERIMENT = 1
INITIAL_EXPERIMENT = 0

config = {"n_games": 250,
          "env":'CartPole-v1'}

def save_data(data, n):

    with open(f"/home/joel/PhD/RL-Skid2Mid/cart_pole/policy_gradients/reinforce{n}_train_data.json", 'w') as outfile:
            json.dump(data, outfile)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    n_games = 500
    i = 0

    train_data = {"train_time": "",
                  "last_epoch": 0,
                  "avg_score": 0}

    n_experiments = N_EXPERIMENT
    
    logger = wandb.init(project="Skid2Mid", config=config)
    for n in range(INITIAL_EXPERIMENT, n_experiments):
        max_avg_score = 0
        agent = Agent(4, 2)

        logger.watch(agent.policy)
        scores, eps_history = [], []
        # for i in range(n_games):
        st = time.time()
        cnt = 0
        while True:
            i += 1
            score = 0
            done = False
            observation = env.reset()
            j = 0
            cnt += 1
            while not done:
                # frame = env.render(mode='rgb_array')

                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                agent.rewards.append(reward)
                # reward = reward_shaping(observation, action)
                score += reward 

                observation = observation_

            actor_loss = agent.learn()
            logger.log({"actor_loss": actor_loss, "score": score}, step=cnt)
            scores.append(score)
            # eps_history.append(agent.epsilion)

            avg_score = np.mean(scores[-300:])
            
            if i % 10 == 0 and avg_score > max_avg_score:
                # logger.unwatch()
                # torch.save(agent.policy.state_dict(), f"/home/joel/PhD/RL-Skid2Mid/cart_pole/policy_gradients/model_reinforce{n}.pt")
                # logger.watch(agent.policy)
                train_data = {"train_time": str(timedelta(seconds=time.time() - st)),
                    "last_epoch": i,
                    "avg_score": avg_score}
                # save_data(train_data, n)
                max_avg_score = avg_score

            print('episode: ', i, 'score %.2f' % score, 
                    'average_score %.2f' % avg_score,
                    f"{round((avg_score / max_avg_score) * 100, 2)}",
                    f'max_avg_score: {max_avg_score}')
            if avg_score > 450 or (avg_score < 0.85 * max_avg_score and avg_score > 100):
                break
        
        train_data = {"train_time": str(timedelta(seconds=time.time() - st)),
                    "last_epoch": i,
                    "avg_score": avg_score}
        save_data(train_data, n)

        x = [i+1 for i in range(n_games)]
        
        logger.unwatch(agent.policy)
        torch.save(agent.policy.state_dict(), f"/home/joel/PhD/RL-Skid2Mid/cart_pole/policy_gradients/model_reinforce{n}.pt")
        del agent