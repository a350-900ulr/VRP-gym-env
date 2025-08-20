from stable_baselines3 import PPO
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from ViennaEnv import ViennaEnv
import numpy as np
from visualizer import Visualizer as Vis
import time
from datetime import datetime
import argparse
from warnings import warn
import re
import os


model_path = 'models/'
def find_best_model(
	desired_model: str,
	pvp: dict
) -> tuple[str, SelfBaseAlgorithm]:
	"""
	At 1st looks for a model with the exact desired parameters. Otherwise, find a model with the
	correct 'pvp' parameters with the highest # of simultaneous trained environments. If
	multiple exists, then use the highest training timestep value. The model name along with the
	model itself is returned.
	"""
	
	
	
	models_found = os.listdir(model_path)
	if desired_model in models_found:
		print(f'Loading model {desired_model}')
		return desired_model, PPO.load(model_path + desired_model)
		
	else:
	
		pattern = re.compile(
			rf'^ppo_vrp_e(\d+)-t(\d+)_pvp'
			rf'-{pvp["place_count"]}'
			rf'-{pvp["vehicle_count"]}'
			rf'-{pvp["package_count"]}\.zip$'
		)
	
		best_file = None
	
		best_e_t = (-1, -1)

		for file_name in models_found:
			match = pattern.match(file_name)
			if match:
				e_val = int(match.group(1))
				t_val = int(match.group(2))
				if (e_val, t_val) > best_e_t:
					best_e_t = (e_val, t_val)
					best_file = file_name
	
		assert best_file is not None, 'No model with correct pvp parameters exists.'
		warn(f'\nLoading best matching model from {best_file}.')
	
		return best_file, PPO.load(model_path + best_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('action', type=str, choices=[
		'train', 'test', 'vis', 'details',
	], nargs='?', default='train')
	parser.add_argument('environment_count', type=int,  nargs='?', default=10)
	parser.add_argument('train_time_k',      type=int,  nargs='?', default=10_000)
	parser.add_argument('place_count',       type=int,  nargs='?', default=80)
	parser.add_argument('vehicle_count',     type=int,  nargs='?', default=5)
	parser.add_argument('package_count',     type=int,  nargs='?', default=10)
	parser.add_argument('verbose',           type=bool, nargs='?', default=False)
	parser.add_argument('verbose_trig_k',    type=int,  nargs='?', default=20_000)
	parser.add_argument('manual_model',      type=str, nargs='?', default=None)

	arguments = parser.parse_args()
	environment_options = {
		'place_count': arguments.place_count,
		'vehicle_count': arguments.vehicle_count,
		'package_count': arguments.package_count,
		'verbose': arguments.verbose,
		'verbose_trigger': arguments.verbose_trig_k * 1_000,
	}
	
	if environment_options['verbose_trigger'] <= arguments.train_time_k * 1_000:
		warn('Verbosity trigger is smaller than training timesteps, so it will likely trigger.')
		
	assert environment_options['package_count'] * 2 <= environment_options['place_count'], \
		'There are less places than unique package origin/target pairs available.'



	# model to write if train is true, model to load if train is false
	
	if arguments.manual_model is None:
		model_name = (
			f'ppo_vrp_e{arguments.environment_count}-t{arguments.train_time_k}_'
			f'pvp'
			f'-{environment_options["place_count"]}'
			f'-{environment_options["vehicle_count"]}'
			f'-{environment_options["package_count"]}'
		)
	else:
		model_name = arguments.manual_model

	match arguments.action:
		case 'train':
			start_time = time.time()
			
			print('training...')
			vec_env = make_vec_env(
				ViennaEnv,
				n_envs=arguments.environment_count,
				env_kwargs=environment_options
			)
			model = PPO('MultiInputPolicy', vec_env, verbose=1)
			model.learn(total_timesteps=arguments.train_time_k * 1_000)
			model.save(model_path + model_name)
			
			results = (
				f'\n{model_name}:'
				f'\n\ttime of training:    {datetime.now().strftime("%y-%m-%d_%H-%M-%S")}'
				f'\n\texecution time:      {time.time() - start_time}'
				f'\n\ttraining time steps: {arguments.train_time_k * 1_000}'
				f'\n\tenvironment count:   {arguments.environment_count}'
				f'\n\tenvironment options:'
				f'\n\t\t* place_count:   {environment_options["place_count"]}'
				f'\n\t\t* vehicle_count: {environment_options["vehicle_count"]}'
				f'\n\t\t* package_count: {environment_options["package_count"]}'
			)
			
			with open(model_path + 'training.log', 'a') as f:
				f.write(results)
			
			print(f'Model saved as {model_name}')

		case 'test':
			print('testing...')
			
			model_name, model = find_best_model(model_name, environment_options)
			vec_env = make_vec_env(
				ViennaEnv,
				n_envs=arguments.environment_count,
				env_kwargs=environment_options
			)
			obs = vec_env.reset()
			done = [False] * arguments.environment_count
			previous_reward = 0

			while not all(list(done)):
				action, _states = model.predict(obs)
				obs, reward, done, info = vec_env.step(action)
				# update progress
				if previous_reward < (current_reward := np.sum(reward)):
					if arguments.environment_count < 4:
						print(
							f'{str(reward):<10}', end='\n' if current_reward % 10 == 0 else ''
						)
					else:
						print(reward)
					previous_reward = current_reward

			final_clock = sum(
				info_single['time'] /
				arguments.environment_count for info_single in info
			)
			total_travel_time = sum(
				info_single['total_travel'] /
				arguments.environment_count for info_single in info
			)

			results = (
				f'\n{model_name}:'
				f'\n\ttime of test:        {datetime.now().strftime("%y-%m-%d_%H-%M-%S")}'
				f'\n\tenvironment count:   {arguments.environment_count}'
				f'\n\tenvironment options:'
				f'\n\t\t* place_count:   {environment_options["place_count"]}'
				f'\n\t\t* vehicle_count: {environment_options["vehicle_count"]}'
				f'\n\t\t* package_count: {environment_options["package_count"]}'
				f'\n\tfinal clock:       {final_clock}'
				f'\n\ttotal travel time: {total_travel_time}'
			)

			print(results)

			with open('testing.log', 'a') as f:
				f.write(results)

			print('\nWrote to test results file.')

		case 'vis':
			
			model_name, model = find_best_model(model_name, environment_options)
	
			vis = Vis(environment_options)
			env = ViennaEnv(**environment_options)
			obs, _ = env.reset()
			done = False

			while not done:
				action, _states = model.predict(obs)
				obs, _, done, _, info = env.step(action)

				vis.draw(info)
				time.sleep(.1)
			
			input('Press ENTER to exit.')

		case 'details':
			_, model = find_best_model(model_path, model_name, environment_options)
			print(model.policy)

		case _:
			raise ValueError('Action must be train, test, or vis.')


