train = True  # run the model.learn() function & save the weights
test = False  # use the model to run an episode

environment_count = 1  # number of simultaneous environments to train on
training_timesteps = 40_000
environment_kwargs = {
	'place_count': 30,
	'vehicle_count': 10,
	'package_count': 10,
	'verbose': False
}

# model to write if train is true, model to load if train is false
model_name = f'ppo_vrp_e{environment_count}-t{training_timesteps}'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv
import numpy as np

if __name__ == '__main__':
	vec_env = make_vec_env(WienEnv, n_envs=environment_count, env_kwargs=environment_kwargs)

	if train:
		model = PPO('MultiInputPolicy', vec_env, verbose=1)
		model.learn(total_timesteps=training_timesteps)
		model.save(model_name)
	else:
		model = PPO.load(model_name)

	if test:
		obs = vec_env.reset()
		done = [False for _ in range(environment_count)]

		previous_reward = 0

		while not all(list(done)):
			action, _states = model.predict(obs)
			obs, reward, done, info = vec_env.step(action)
			# update progress
			if previous_reward < (current_reward := np.sum(reward)):
				print(reward, end='\n' if current_reward % 10 == 0 else '')
				previous_reward = current_reward

		print(info)

