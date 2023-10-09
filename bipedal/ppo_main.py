import os
import gym
from agents.ppo import Agent
import wandb
import time
import numpy as np


def watch(logger: wandb, agent: Agent):
    logger.watch(agent.actor )
    logger.watch(agent.critic)

def unwatch(logger: wandb, agent: Agent):
    logger.unwatch(agent.actor )
    logger.unwatch(agent.critic)


config = {"chkpt_dir":  "/home/joel/PhD/results/rl/models",
          "n_games": 2000,
          "N": 500,
          "env":'BipedalWalker-v3',
          "experiment": "baseline1",
          "batch_size": 100,
          "fc1_dims": 1024,
          "fc2_dims": 512,
          "n_epochs": 2,
          "policy_clip": 0.1,
          "alpha": 0.0003,
          "beta": 0.001,
          "action_std_init": 0.6,
          "action_std_decay_rate": 0.05,
          "min_action_std": 0.05,
          "gae_lambda": 0.9,
          "algorithm": "ppo",
          "debug": True}


# config = {"n_games": 200000,
#           "N": 500,
#           "env":'BipedalWalker-v3',
#           "chkpt_dir":  "/root/models",
#           "experiment": "baseline9",
#           "batch_size": 50,
#           "fc1_dims": 1024,
#           "fc2_dims": 512,
#           "n_epochs": 2,
#           "alpha": 0.0003,
#           "beta": 0.001,
#           "policy_clip": 0.1,
#           "action_std_init": 0.6,
#           "action_std_decay_rate": 0.005,
#           "min_action_std": 0.1,
#           "gae_lambda": 0.5,
#           "algorithm": "ppo",
#           "debug": False}


if __name__== "__main__":
    print(config)
    chkpt_dir = os.path.join(config["chkpt_dir"], config["algorithm"] ,config["experiment"])
    alg_dir = os.path.join(config["chkpt_dir"], config["algorithm"])
    
    if not os.path.isdir(alg_dir):
        os.mkdir(alg_dir)

    if not os.path.isdir(chkpt_dir):
        os.mkdir(chkpt_dir)

    env = gym.make(config["env"])
    

    if not config["debug"]: logger = wandb.init(project="Bipedal", config=config, name=f"{config['algorithm']}_{config['experiment']}")
    
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.shape, env=env,n_epochs= config["n_epochs"],
                  alpha=config["alpha"], beta=config["beta"], batch_size=config["batch_size"], action_std_decay=config["action_std_decay_rate"], 
                  action_std_min=config["min_action_std"], chkpt_dir=chkpt_dir, action_std_init=config["action_std_init"], 
                  fc1=config["fc1_dims"], fc2=config["fc2_dims"] ,gae_lambda=config["gae_lambda"], policy_clip=config["policy_clip"])

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
        stillness = 0
        while not done:
            cnt += 1

            action, prob, val = agent.choose_action(observation)

            observation_, reward, done, unkwon, info = env.step(action)

            if abs(observation[2]) < 0.02:
                stillness += 1
                if stillness >= 10:
                    done = True
                    reward = -100
            else:
                stillness = 0

            if done or unkwon: 
                done = True
                # if score <= 450: reward = -20

            n_steps += 1
            score += reward
            
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % config["N"] == 0:
                actor_loss, critic_loss = agent.learn()
                learn_iters += 1
                agent.actor.set_action_std()
                if not config["debug"]: 
                    logger.log({"iter": learn_iters, 
                                "actor_loss": actor_loss,
                                "critic_loss": critic_loss}, step=cnt)

            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-25:])

        if not config["debug"]: 
            logger.log({"episode": i, 
                        "time": time.time() - st, 
                        "score": score}, step=cnt)

        if avg_score > best_score:
            best_score = avg_score
            if not config["debug"]: agent.save_models()

        print(f'episode {i}, score {score}, avg score {avg_score}, learning_steps {learn_iters}, agent_std {agent.actor.action_std}')


