import gym
from dq_learning import Agent
from utils import plotLearning
import numpy as np
import torch
import imageio
import os
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    

def _label_with_episode_number(frame, reward):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Reward: {reward}', fill=text_color)

    return im


if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilion=0.00, batch_size=256, n_actions=4,
    eps_end=0.01, input_dims=[8], lr=0.0001)

    print(agent.Q_eval.device)

    agent.Q_eval.load_state_dict(torch.load("/home/joel/PhD/RL-Skid2Mid/model.pt", map_location=torch.device("cpu")))
    agent.Q_eval.eval()

    scores, eps_history = [], []
    n_games = 1

    done = False
    score = 0
    observation = env.reset()

    frames = []

    while not done:

        frame = env.render(mode='rgb_array')
        frames.append(_label_with_episode_number(frame, score))

        action = agent.predict_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward 

        observation = observation_
    
    env.close()
    imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)
    print("Done!")
