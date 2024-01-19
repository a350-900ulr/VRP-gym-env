# Main options
train = False  # run the model.learn() function & save the weights
test = True  # use the model to run an episode
visualize = True  # display actions in the environment

# Training options. Otherwise, the model loaded for testing using the default naming format.
# To load a model with a different name, change the model_name variable manually.
environment_count = 1  # number of simultaneous environments to train on
training_timesteps_k = 1  # max number of iterations to train on (multiplied by 1,000)

# Environment options
environment_options = {
	'place_count': 5,
	'vehicle_count': 1,
	'package_count': 2,
	'verbose': False,  # print out vehicle & package info during each `step()`
	# if verbose is False, activate verbosity anyway after this many steps.
	# this is useful if the model gets stuck.
	'verbose_trigger': 100_000
}

# model to write if train is true, model to load if train is false
model_name = (
	f'ppo_vrp_e{environment_count}-t{training_timesteps_k}_'
	f'pvp'
	f'-{environment_options["place_count"]}'
	f'-{environment_options["vehicle_count"]}'
	f'-{environment_options["package_count"]}'
)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv
import numpy as np
import os
from src.visualizer import Visualizer as Vis
import time
import warnings
from datetime import datetime


if __name__ == '__main__':
	if train:
		vec_env = make_vec_env(WienEnv,
			n_envs = environment_count,
			env_kwargs = environment_options
		)
		print('training...')
		model = PPO('MultiInputPolicy', vec_env, verbose=1)
		model.learn(total_timesteps=training_timesteps_k * 1_000)
		model.save(model_name)
		print(f'Model saved as {model_name}')
	else:
		model = PPO.load(model_name)
		print(f'Loaded model {model_name}')

	if test:
		print('testing...')
		vec_env = make_vec_env(WienEnv, n_envs=1, env_kwargs=environment_options)
		obs = vec_env.reset()
		done = [False for _ in range(environment_count)]

		if visualize:
			vis = Vis(environment_options)
			if environment_count != 1:
				warnings.warn(
					'Only 1 environment can be visualized at a time, environment_count set to 1'
				)
				environment_count = 1

		previous_reward = 0

		while not all(list(done)):
			action, _states = model.predict(obs)
			obs, reward, done, info = vec_env.step(action)
			# update progress
			if previous_reward < (
					current_reward := np.sum(reward) - environment_options['package_count']
			):
				if environment_count < 4:
					print(f'{str(reward):<10}', end='\n' if current_reward % 10 == 0 else '')
				else:
					print(reward)
				previous_reward = current_reward

			if visualize:
				vis.draw(info[0])
				time.sleep(.2)

		with open('results.txt', 'a') as f:
			f.write(f'\n{model_name}:')
			f.write(f'\n\ttime of test:      {datetime.now().strftime("%y-%m-%d_%H-%M-%S")}')
			f.write(f'\n\tenvironment options:')
			f.write(f'\n\t\tplace_count:    {environment_options["place_count"]}')
			f.write(f'\n\t\tvehicle_count:  {environment_options["vehicle_count"]}')
			f.write(f'\n\t\tpackage_count:  {environment_options["package_count"]}')
			f.write(f'\n\tfinal time:        {info[0]["time"]}')
			f.write(f'\n\ttotal travel time: {info[0]["total_travel"]}')

		f.close()

	print('\nWrote to results file')
