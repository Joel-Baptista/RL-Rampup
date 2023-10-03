import gym
import numpy as np
from agents.ddpg import Agent
import torch
from wandb_logs import WandbLogger
import wandb
import time

def watch(logger, agent):
    logger.watch(agent.actor, )
    # logger.watch(agent.target_actor)
    logger.watch(agent.critic)
    # logger.watch(agent.target_critic)


def unwatch(logger, agent):
    logger.unwatch(agent.actor)
    logger.unwatch(agent.target_actor)
    logger.unwatch(agent.critic)
    logger.unwatch(agent.target_critic)

config = {"n_games": 1000,
          "env":'Pendulum-v0'}

if __name__== "__main__":

    env = gym.make(config["env"])
    

    logger = wandb.init(project="Skid2Mid", config=config)
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape,
                    chkpt_dir="")


    watch(logger, agent)
    n_games = config["n_games"]

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps=0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)

            n_steps += 1

        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    
    st = time.time()
    cnt = 0
    for i in range(n_games):
        
        
        observation = env.reset()
        done = False
        score = 0

        while not done:           
            cnt += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            logger.log({"reward": reward}, step=cnt)
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            if not load_checkpoint:
                actor_loss, critic_loss = agent.learn()
                logger.log({"actor_loss": actor_loss, "critic_loss": critic_loss}, step=cnt)

            observation = observation_

        logger.log({"episode": i, "time": time.time() - st, "score": score}, step=cnt)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                unwatch(logger, agent)
                agent.save_models()
                watch(logger, agent)

        print(f'episode {i}, score {score}, avg score {avg_score}')


