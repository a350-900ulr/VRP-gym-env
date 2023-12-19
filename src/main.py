input_model = "ppo_vrp1_50"
output_model = None

'''
https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl#50663200

ideas: limit time riding (as episode), make part of reward distance packet is from target
'''

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.wien_env import WienEnv

if __name__ == '__main__':
	if input_model is None and output_model is None:
		assert False, print('no models defined')
	elif input_model is not None and output_model is not None:
		assert False, print('cannot define both input & output')
	elif output_model is not None:
		vec_env = make_vec_env(WienEnv, n_envs=1)

		model = PPO('MultiInputPolicy', vec_env, verbose=1)
		model.learn(total_timesteps=50000)
		model.save(output_model)

	elif input_model is not None:
		vec_env = make_vec_env(WienEnv, n_envs=1)
		model = PPO.load(input_model)

		obs = vec_env.reset()
		done = False
		while not done:
			action, _states = model.predict(obs)
			obs, reward, done, info = vec_env.step(action)
			if 0 < reward: print(reward, end='')
			vec_env.render("human")
