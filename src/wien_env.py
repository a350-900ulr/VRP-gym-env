# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

import gymnasium as gym
from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance, MultiBinary
from wien_graph import WienGraph
import random

class CustomEnv(gym.Env):
	def __init__(self, place_count: int, vehicle_count: int, package_count: int):
		super().__init__()

		# initial values for environment itself
		def filler(amount, fill_with=None, random_up_to_placeholder=False):
			if random_up_to_placeholder:
				return [random.randrange(fill_with) for _ in range(amount)]
			else:
				return [fill_with for _ in range(amount)]

		self.vehicles = {
			'id': range(vehicle_count),
			'available': filler(vehicle_count, True),
			'start': filler(vehicle_count, place_count, True),
			'end': filler(vehicle_count),
			'progress': filler(vehicle_count, 0),
			'package': filler(vehicle_count)
		}

		self.packages = {
			'id': range(package_count),
			'location_current': filler(package_count, place_count, True),
			'location_target':  filler(package_count, place_count, True),
			'carrying_vehicle': filler(package_count),
		}
		############# i was here

		self.observation_space = Dict({
			'distance_matrix': WienGraph().create_instance(),  # GraphInstance
			'vehicle information': Dict({
				'id': Discrete(vehicle_count),
				'availability': MultiBinary(vehicle_count),
				'transit_start': Discrete(vehicle_count),
				'transit_end': Discrete(vehicle_count),
				'transit_progress': Discrete(vehicle_count),
				'package_id': Discrete(vehicle_count),
			}),
			'package_information': Dict({
				'id': Discrete(package_count),
				'location_start': Discrete(package_count),
				'location_target':  Discrete(package_count),
				'location_current': Discrete(package_count),
				'carrying_vehicle': Discrete(package_count)
			}),
		})

		# possible values are in the range of the number of locations
		self.action_space = Discrete(vehicle_count)


	def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		"""
		Run one timestep of the environment’s dynamics using the agent actions.

		:param action: idk man
		:return: lots of stuff
		"""

		observation (ObsType) – An element of the environment’s observation_space as the next observation due to the agent actions. An example is a numpy array containing the positions and velocities of the pole in CartPole.

		return observation, reward, terminated, truncated, info

	def reset(self, seed=None, options=None):
		...
		return observation, info

	def render(self):
		...

	def close(self):
		...