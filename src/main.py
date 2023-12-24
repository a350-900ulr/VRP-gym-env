# TODO: create datetime in file name, actually train model

train = True  # run the model.learn() function & save the weights
test = True  # use the model to run an episode

environment_count = 1  # number of simultaenous environments to train on
training_timesteps = 50
environment_kwargs = {
	'place_count': 30,
	'vehicle_count': 10,
	'package_count': 10,
	'verbose': False}


# model to write if train is true, model to load if train is false
model_name = f'test_ppo_vrp_e{environment_count}-t{training_timesteps}'

'''
https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl#50663200
'''

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
	vec_env = make_vec_env(WienEnv, n_envs=environment_count, env_kwargs=environment_kwargs)

	if train:
		model = PPO('MultiInputPolicy', vec_env, verbose=1)
		model.learn(total_timesteps=training_timesteps)
		model.save(model_name)

	if test:
		if not train:
			model = PPO.load(model_name)

		obs = vec_env.reset()
		done = False

		pbar = tqdm(total=environment_kwargs['package_count'] * environment_count)
		while not done:

			action, _states = model.predict(obs)
			obs, reward, done, info = vec_env.step(action)

			#print(reward, end='-')
			#vec_env.render()
			pbar.update(np.sum(reward))
		pbar.close()
