# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

import gymnasium as gym
from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance, MultiBinary
from src.wien_graph import WienGraph
import random

class CustomEnv(gym.Env):

	"""Custom Environment that follows gym interface."""
	metadata = {"render_modes": ["human"], "render_fps": 30}

	def __init__(self, place_count: int, vehicle_count: int, package_count: int):
		super().__init__()

		self.clock = 0
		self.vehicle_count = vehicle_count
		self.package_count = package_count

		def filler(amount: int, fill_with=None, random_int_up_to_fill: bool=False):
			"""
			:param amount: length of list to return
			:param fill_with: value to fill list with, or in case of randomization an integer in range [0, fill_with)
			:param random_int_up_to_fill: choice to randomize an integer range
			:return: list filled with specified values
			"""
			if random_int_up_to_fill:
				return [random.randrange(fill_with) for _ in range(amount)]
			else:
				return [fill_with for _ in range(amount)]

		# initial values for environment itself. This will also be returned during self.step()
		self.environment = {
			'distances': WienGraph().create_instance(),  # GraphInstance
			'vehicles': {
				'id': range(vehicle_count),
				'available': filler(vehicle_count, True),
				'start': filler(vehicle_count, place_count, True),
				'end': filler(vehicle_count),
				'progress': filler(vehicle_count, 0),
				'package': filler(vehicle_count)
			},
			'packages': {
				'id': range(package_count),
				'location_current': filler(package_count, place_count, True),
				'location_target':  filler(package_count, place_count, True),
				'carrying_vehicle': filler(package_count),
			},
		}


		self.observation_space = Dict({
			'distance_info': WienGraph(),  # GraphInstance
			'vehicle_info': Dict({
				'id': Discrete(vehicle_count),
				'availability': MultiBinary(vehicle_count),
				'transit_start': Discrete(vehicle_count),
				'transit_end': Discrete(vehicle_count),
				'transit_progress': Discrete(vehicle_count),
				'package_id': Discrete(vehicle_count),
			}),
			'package_info': Dict({
				'id': Discrete(package_count),
				'location_current': Discrete(package_count),
				'location_target':  Discrete(package_count),
				'carrying_vehicle': Discrete(package_count)
			}),
		})

		# possible values are in the range of the number of locations
		self.action_space = Discrete(vehicle_count)


	def step(self, action) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
		"""
		Run one timestep of the environment’s dynamics using the agent actions.

		:param action: idk man
		:return: lots of stuff
		"""
		assert action.size == self.vehicle_count
		self.clock += 1

		for vehicle_decision, index in enumerate(action):
			if self.environment['vehicles']['available'][index]:

				...
		observation (ObsType) – An element of the environment’s observation_space as the next observation due to the agent actions. An example is a numpy array containing the positions and velocities of the pole in CartPole.

		return observation, reward, terminated, truncated, info

	def reset(self, seed=None, options=None):
		...
		return observation, info

	def render(self):
		...

	def close(self):
		...

	def automate_pickup_and_dropoff(self):
		for i in range(self.package_count):
			current_package = self.environment['packages']
			if (self.environment['packages']['location_current'] == ['location_target'])
			# how to check if package is delivered?

			'location_current': filler(package_count, place_count, True),
			'location_target':  filler(package_count, place_count, True),