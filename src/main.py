## Main options
train = False  # run the model.learn() function & save the weights
test = True  # use the model to run an episode

## Training options
environment_count = 1  # number of simultaneous environments to train on
training_timesteps = 10_000  # total number of samples (env steps) to train on

## Environment options
environment_options = {
	'place_count': 30,
	'vehicle_count': 10,
	'package_count': 10,
	'verbose': True,  # print out vehicle & package info during each `step()`
	# if verbose is False, activate verbosity anyway after this many steps.
	# this is useful if the model gets stuck.
	'verbose_trigger': 100_000
}

# model to write if train is true, model to load if train is false
model_name = f'ppo_vrp_e{environment_count}-t{training_timesteps}'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv
import numpy as np
import os

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
	vec_env = make_vec_env(WienEnv, n_envs=environment_count, env_kwargs=environment_options)

	if train:
		print('training...')
		model = PPO('MultiInputPolicy', vec_env, verbose=1)
		model.learn(total_timesteps=training_timesteps)
		model.save(model_name)
		print(f'Model saved as {model_name}')
	else:
		model = PPO.load(model_name)
		print(f'Loaded model {model_name}')

	if test:
		print('testing...')
		obs = vec_env.reset()
		done = [False for _ in range(environment_count)]

		previous_reward = 0

		while not all(list(done)):
			action, _states = model.predict(obs)
			obs, reward, done, info = vec_env.step(action)
			# update progress
			if previous_reward < (current_reward := np.sum(reward)):

				#print(f'{str(reward):<10}', end='\n' if current_reward % 10 == 0 else '')
				previous_reward = current_reward
		print(info)
