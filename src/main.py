from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wien_env import WienEnv
import numpy as np
from visualizer import Visualizer as Vis
import time
from datetime import datetime
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('action', type=str)
	for int_arg in [
		'-environment_count', '-training_timesteps_k',
		'-place_count', '-vehicle_count', '-package_count'
	]:
		parser.add_argument(int_arg, type=int, required=False)
	parser.add_argument('-verbose', type=bool, required=False)
	parser.add_argument('-verbose_trigger', type=int, required=False)

	arguments = parser.parse_args()

	environment_count = \
		arguments.environment_count if arguments.environment_count is not None else \
		10
	training_timesteps_k = \
		arguments.training_timesteps_k if arguments.training_timesteps_k is not None else \
		100
	environment_options = {
		'place_count':
			arguments.place_count if arguments.place_count is not None else
			80,
		'vehicle_count':
			arguments.vehicle_count if arguments.vehicle_count is not None else
			10,
		'package_count':
			arguments.package_count if arguments.package_count is not None else
			20,
		'verbose':
			arguments.verbose if arguments.verbose is not None else
			False,
		'verbose_trigger':
			arguments.verbose_trigger if arguments.verbose_trigger is not None else
			100_000,
	}

	# model to write if train is true, model to load if train is false
	model_name = (
		f'ppo_vrp_e{environment_count}-t{training_timesteps_k}_'
		f'pvp'
		f'-{environment_options["place_count"]}'
		f'-{environment_options["vehicle_count"]}'
		f'-{environment_options["package_count"]}'
	)


	match arguments.action:
		case 'train':
			print('training...')
			vec_env = make_vec_env(
				WienEnv,
				n_envs = environment_count,
				env_kwargs = environment_options
			)
			model = PPO('MultiInputPolicy', vec_env, verbose=1)
			model.learn(total_timesteps=training_timesteps_k * 1_000)
			model.save(model_name)
			print(f'Model saved as {model_name}')

		case 'test':
			print('testing...')
			model = PPO.load(model_name)
			print(f'Loaded model {model_name}')

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
				if previous_reward < (current_reward := np.sum(reward)):
					if environment_count < 4:
						print(
							f'{str(reward):<10}', end='\n' if current_reward % 10 == 0 else ''
						)
					else:
						print(reward)
					previous_reward = current_reward

			final_clock = sum(info_single['time']/environment_count for info_single in info)
			total_travel_time = sum(
				info_single['total_travel']/environment_count for info_single in info
			)

			results = (
				f'\n{model_name}:'
				f'\n\ttime of test:      {datetime.now().strftime("%y-%m-%d_%H-%M-%S")}'
				f'\n\tenvironment count: {environment_count}'
				f'\n\tenvironment options:'
				f'\n\t\t* place_count:   {environment_options["place_count"]}'
				f'\n\t\t* vehicle_count: {environment_options["vehicle_count"]}'
				f'\n\t\t* package_count: {environment_options["package_count"]}'
				f'\n\tfinal clock:       {final_clock}'
				f'\n\ttotal travel time: {total_travel_time}'
			)

			print(results)

			with open('results.txt', 'a') as f:
				f.write(results)

			f.close()

			print('\nWrote to test results file')

		case 'visualize':
			model = PPO.load(model_name)
			print(f'Loaded model {model_name}')

			vis = Vis(environment_options)
			env = WienEnv(**environment_options)
			obs, _ = env.reset()
			done = False

			while not done:
				action, _states = model.predict(obs)
				obs, _, done, _, info = env.step(action)

				vis.draw(info)
				time.sleep(.2)

		case _:
			raise ValueError('action must be train, test, or visualize')