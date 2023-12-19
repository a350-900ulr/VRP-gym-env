'''
https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl#50663200

ideas: limit time riding (as episode), make part of reward distance packet is from target
'''

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv

vec_env = make_vec_env(WienEnv, n_envs=1)
model = PPO('MultiInputPolicy', vec_env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_vrp1_50")




model = PPO.load("ppo_vrp1_50")

obs = vec_env.reset()
done = False
while not done:
	action, _states = model.predict(obs)
	obs, rewards, done, info = vec_env.step(action)
	print(rewards, end='')
	vec_env.render("human")

