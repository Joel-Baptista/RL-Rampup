import os
import gym
from agents.td3 import Agent
import wandb
import time
import numpy as np


def watch(logger: wandb, agent: Agent):
    logger.watch(agent.actor )
    logger.watch(agent.critic_1)
    logger.watch(agent.critic_2)

def unwatch(logger: wandb, agent: Agent):
    logger.unwatch(agent.actor )
    logger.unwatch(agent.critic_1)
    logger.unwatch(agent.critic_2)


config = {"n_games": 2000,
          "env":'BipedalWalker-v3',
          "chkpt_dir": "/root/models",
          "experiment": "baseline",
          "alpha": 0.001,
          "algorithm": "td3",
          "beta": 0.001,
          "batch_size": 100,
          "fc1": 1024,
          "fc2": 512,
          "debug": False}

# config = {"n_games": 2000,
#           "env":'BipedalWalker-v3',
#           "chkpt_dir": "/home/joel/PhD/results/rl/models",
#           "algorithm": "td3",
#           "alpha": 0.001,
#           "beta": 0.001,
#           "batch_size": 100,
#           "experiment": "baseline",
#           "fc1": 1024,
#           "fc2": 512,
#           "debug": True}

if __name__== "__main__":

    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    alg_dir = os.path.join(config["chkpt_dir"], config["algorithm"])
    

    if not os.path.isdir(alg_dir):
        os.mkdir(alg_dir)

    if not os.path.isdir(chkpt_dir):
        os.mkdir(chkpt_dir)

    env = gym.make(config["env"])
    

    if not config["debug"]: logger = wandb.init(project="Bipedal", config=config)
    
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape,
                  fc1=config["fc1"], fc2=config["fc2"], alpha=config["alpha"], beta=config["beta"], batch_size=config["batch_size"],
                    chkpt_dir=chkpt_dir)

    if not config["debug"]: watch(logger, agent)

    st = time.time()
    cnt = 0
    score_history = []
    best_score = env.reward_range[0]
    for i in range(config["n_games"]):
        
        
        observation, _ = env.reset()
        done = False
        score = 0
        max_speed = 0
        speed = []

        while not done:           
            cnt += 1
            
            action = agent.choose_action(observation)

            observation_, reward, done, unkown, info = env.step(action)

            if observation_[2] > max_speed: max_speed = observation_[2]
            speed.append(observation_)

            if done or unkown: done = True
    
            if not config["debug"]: logger.log({"reward": reward}, step=cnt)
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            
            actor_loss, critic1_loss, critic2_loss  = agent.learn()

            if (critic1_loss is not None or critic2_loss is not None) and not config["debug"]:
                logger.log({"critic_loss": critic1_loss, "critic2_loss": critic2_loss}, step=cnt)

            if actor_loss is not None  and not config["debug"]:
                logger.log({"actor_loss": actor_loss}, step=cnt)

            observation = observation_

        if not config["debug"]: 
            logger.log({"episode": i, 
                        "time": time.time() - st, 
                        "score": score, 
                        "eps_max_speed": max_speed, 
                        "eps_avg_speed": np.mean(speed)}, step=cnt)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i}, score {score}, avg score {avg_score}')

    env.close()