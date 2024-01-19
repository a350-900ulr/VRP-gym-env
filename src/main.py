# Main options
train = False  # run the model.learn() function & save the weights
test = True  # use the model to run an episode
visualize = False  # display actions in the environment

# Training options. Otherwise, the model loaded for testing using the default naming format.
# To load a model with a different name, change the model_name variable manually.
environment_count = 10  # number of simultaneous environments to train on
training_timesteps_k = 10  # max number of iterations to train on (multiplied by 1,000)

# Environment options
environment_options = {
	'place_count': 80,
	'vehicle_count': 10,
	'package_count': 20,
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
from src.visualizer import Visualizer as Vis
import time
import warnings
from datetime import datetime


if __name__ == '__main__':
	if train:
		vec_env = make_vec_env(
			WienEnv,
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

		if visualize:  # only 1 environment can be visualized
			vis = Vis(environment_options)
			env = WienEnv(**environment_options)
			obs, _ = env.reset()
			done = False

			while not done:
				action, _states = model.predict(obs)
				obs, _, done, _, info = env.step(action)

				if visualize:
					vis.draw(info)
					time.sleep(.2)

			final_clock = info['time']
			total_travel_time = info['total_travel']

		else:  # otherwise use the desired amount of environments
			vec_env = make_vec_env(
				WienEnv,
				n_envs = environment_count,
				env_kwargs = environment_options
			)
			obs = vec_env.reset()
			done = [False] * environment_count
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

			final_clock = sum(info_single['time']/environment_count for info_single in info)
			total_travel_time = sum(info_single['total_travel']/environment_count for info_single in info)

		with open('results.txt', 'a') as f:
			f.write(f'\n{model_name}:')
			f.write(f'\n\ttime of test:      {datetime.now().strftime("%y-%m-%d_%H-%M-%S")}')
			f.write(f'\n\tenvironment count: {environment_count}')
			f.write(f'\n\tenvironment options:')
			f.write(f'\n\t\t* place_count:   {environment_options["place_count"]}')
			f.write(f'\n\t\t* vehicle_count: {environment_options["vehicle_count"]}')
			f.write(f'\n\t\t* package_count: {environment_options["package_count"]}')
			f.write(f'\n\tfinal clock:       {final_clock}')
			f.write(f'\n\ttotal travel time: {total_travel_time}')

		f.close()

		print('\nWrote to test results file')
