import gym
import numpy as np
from agents.sac import Agent
import torch
from wandb_logs import WandbLogger
import wandb
import time
import os

def watch(logger: wandb, agent: Agent):
    logger.watch(agent.actor )
    logger.watch(agent.critic_1)
    logger.watch(agent.critic_2)

def unwatch(logger: wandb, agent: Agent):
    logger.unwatch(agent.actor )
    logger.unwatch(agent.critic_1)
    logger.unwatch(agent.critic_2)


# config = {
#           "n_games": 2000,
#         #   "env":'BipedalWalker-v3',
#           "env":'MountainCarContinuous-v0',
#           "chkpt_dir": "/home/joel/PhD/results/rl/models",
#           "experiment": "baseline5",
#           "algorithm": "sac",
#           "reward_scale": 1,
#           "alpha": 0.0003,
#           "beta": 0.003,
#           "fc1": 256,
#           "fc2": 256,
#           "batch_size": 1,
#           "debug": True}

config = {"n_games": 3000,
          "env":'BipedalWalker-v3',
          "chkpt_dir":  "/root/models",
          "experiment": "baseline8",
          "algorithm": "sac",
          "reward_scale": 100,
          "alpha": 0.0003,
          "beta": 0.003,
          "fc1": 1024,
          "fc2": 512,
          "batch_size": 256,
          "debug": False}

if __name__== "__main__":
    
    print(config)
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    alg_dir = os.path.join(config["chkpt_dir"], config["algorithm"])
    
    if not os.path.isdir(alg_dir):
        os.mkdir(alg_dir)

    if not os.path.isdir(chkpt_dir):
        os.mkdir(chkpt_dir)

    # env = gym.make(config["env"])
    env = gym.make(config["env"], render_mode="human")
    

    if not config["debug"]: logger = wandb.init(project="Bipedal", config=config, name=f"{config['algorithm']}_{config['experiment']}")
    
    if not config["debug"]: logger = wandb.init(project="Bipedal", config=config)
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape,
                  fc1=config["fc1"], fc2=config["fc2"], alpha=config["alpha"], beta=config["beta"], batch_size=config["batch_size"],
                    chkpt_dir=chkpt_dir, reward_scale=config["reward_scale"])


    if not config["debug"]: watch(logger, agent)

    
    st = time.time()
    cnt = 0
    score_history = []
    best_score = env.reward_range[0]
    n_steps=0
    learn_iters = 0

    for i in range(config["n_games"]):
        observation, _ = env.reset()
        done = False
        score = 0.0

        while not done:           
            cnt += 1
            
            action = agent.choose_action(observation)
            
            observation_, reward, done, unkown, info = env.step(action)

            if done or unkown: done = True
    
            if not config["debug"]: logger.log({"reward": reward}, step=cnt)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            actor_loss, critic1_loss, critic2_loss, value_loss, log_probs = agent.learn()


            if not config["debug"]:
                logger.log({"actor_loss": actor_loss,
                            "critic1_loss": critic1_loss,
                            "critic2_loss": critic2_loss,
                            "value_loss": value_loss,
                            "log_probs": log_probs}, step=cnt)

            observation = observation_

        if not config["debug"]: 
            logger.log({"episode": i, 
                        "time": time.time() - st, 
                        "score": score}, step=cnt)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not config["debug"]: agent.save_models()

        print(f'episode {i}, score {score}, avg score {avg_score}')

    env.close()

