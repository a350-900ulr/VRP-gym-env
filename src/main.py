places = 10  # locations to use, from 0-80
packages = 200
vehicles = 20

'''
https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl#50663200

ideas: limit time riding (as episode), make part of reward distance packet is from target
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time

# distance matrix















import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv

vec_env = make_vec_env(WienEnv, n_envs=4)
model = PPO('MultiInputPolicy', vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_vrp")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
	action, _states = model.predict(obs)
	obs, rewards, done, info = vec_env.step(action)
	print(rewards, info)
	vec_env.render("human")

