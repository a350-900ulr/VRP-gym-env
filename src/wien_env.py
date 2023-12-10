# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance
from wien_graph import WienGraph, WienGraphArgs

class CustomEnv(gym.Env):
	def __init__(self, vehicle_count: int):
		super().__init__()


		self.observation_space = Tuple(
			# distance matrix itself
			WienGraph().create_instance(),
			# vehicle information
			Dict({"position": Discrete(2), "velocity": Discrete(3)}),
			# package information
			Dict({})
		)



		self.action_space = MultiDiscrete([places + 1] * vehicle_count)


	def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		"""
		Run one timestep of the environment’s dynamics using the agent actions.

		:param action: idk man
		:return: lots of stuff
		"""
		return observation, reward, terminated, truncated, info

	def reset(self, seed=None, options=None):
		...
		return observation, info

	def render(self):
		...

	def close(self):
		...