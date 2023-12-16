from typing import Any

import gymnasium as gym
from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance, MultiBinary
from src.wien_graph import WienGraph
from src.funcs import create_distance_matrix, filler
import random

class WienEnv(gym.Env):

	"""Custom Environment that follows gym interface."""
	metadata = {"render_modes": ["human"], "render_fps": 30}

	def __init__(self, place_count: int = 20, vehicle_count: int = 10, package_count: int = 10):
		super().__init__()

		self.clock = 0
		self.place_count = place_count
		self.vehicle_count = vehicle_count
		self.package_count = package_count
		self.total_travel = 0
		self.graph_object = WienGraph(20)
		self.place_indices = self.graph_object.picks
		self.distance_matrix = create_distance_matrix()

		# initial values for environment itself. This will also be returned during self.step()
		self.environment = self.reset()

		self.observation_space = Dict({
			'distance_info': WienGraph().template,
			'vehicle_info': Dict({
				'id': Discrete(vehicle_count),
				'availability': MultiBinary(vehicle_count),
				'transit_start': Discrete(vehicle_count),
				'transit_end': Discrete(vehicle_count),
				'transit_remaining': Discrete(vehicle_count),
				'has_package': Discrete(vehicle_count),
			}),
			'package_info': Dict({
				'id': Discrete(package_count),
				'location_current': Discrete(package_count),
				'location_target':  Discrete(package_count),
				'carrying_vehicle': Discrete(package_count),
				'delivered': MultiBinary(package_count),
			}),
		})

		# possible values are in the range of the number of locations
		self.action_space = Discrete(vehicle_count)

	def step(self, action) -> tuple[dict, int, bool, bool, dict[str, Any]]:
		"""
		Run one timestep of the environment’s dynamics using the agent actions.

		:param action: idk man
		:return: lots of stuff
		"""
		assert action.size == self.vehicle_count
		for dispatch_location in action:
			assert dispatch_location in self.place_indices
		self.clock += 1

		for vehicle_decision, v in enumerate(action):
			def vehi(key): return self.environment['vehicles'][key][v]
			def vehi_set(key, val): self.environment['vehicles'][key][v] = val

			if vehi('transit_remaining') == 0:
				vehi_set('transit_end', vehicle_decision)
				vehi_set('transit_remaining',
					self.distance_matrix[vehi('transit_start')][vehicle_decision]
				)

		# progress each vehicle
		for v in range(self.vehicle_count):
			if self.environment['vehicles']['transit_remaining'][v] > 0:
				self.environment['vehicles']['transit_remaining'][v] -= 1
				self.total_travel += 1

		return (
			self.environment,
			self.automate_packages(),  # drop off & pick up any packages
			self.all_delivered(),
			False,
			{'time': self.clock},
		)

	def reset(self, seed=None, options=None) -> dict:
		# y does this needa return anything?
		return {
			'distances': self.graph_object.create_instance(),  # GraphInstance
			'vehicles': {
				'id': range(self.vehicle_count),
				'available': filler(self.vehicle_count, True),
				'transit_start': filler(
					self.vehicle_count, self.place_count, True),
				'transit_end': filler(self.vehicle_count),
				'transit_remaining': filler(self.vehicle_count, 0),
				'has_package': filler(self.vehicle_count)
			},
			'packages': {
				'id': range(self.package_count),
				'location_current': filler(
					self.package_count, self.place_count, True),
				'location_target':  filler(
					self.package_count, self.place_count, True),
				'carrying_vehicle': filler(self.package_count),
				'delivered': filler(self.package_count, False)
			},
		}

	def render(self):
		...

	def close(self):
		...

	def automate_packages(self):
		"""
		automatically pick up packages when vehicle passes over, automatically drop off at location
		:return: number of packages that have been delivered
		"""
		reward = 0
		for p in range(self.package_count):

			# make sure delivered statuses are updated
			def pack(key): return self.environment['packages'][key][p]
			def pack_set(key, val): self.environment['packages'][key][p] = val
			if pack('location_current') == pack('location_target') and not pack('delivered'):
				self.environment['packages']['delivered'][p] = True

			if not pack('delivered') and pack('carrying_vehicle') is None:
				# make sure vehicles fulfill any request at any location if possible
				for v in range(self.vehicle_count):
					def vehi(key): return self.environment['vehicles'][key][v]
					def vehi_set(key, val): self.environment['vehicles'][key][v] = val
					if (vehi('available') and
						vehi('transit_start') == pack('location_current') and
						vehi('transit_remaining') == 0
					):
						vehi_set('available', False)
						vehi_set('has_package', p)
						pack_set('carrying_vehicle', v)

			elif not pack('delivered') and pack('carrying_vehicle') is not None:
				def vehi(key):
					return self.environment['vehicles'][key][pack('carrying_vehicle')]

				def vehi_set(key, val):
					self.environment['vehicles'][key][pack('carrying_vehicle')] = val
				# check if package could be delivered
				if (vehi('transit_start') == pack('location_target') and
					vehi('transit_remaining') == 0
				):
					pack_set('carrying_vehicle', None)
					pack_set('delivered', True)
					vehi_set('available', True)
					vehi_set('has_package', None)
					reward += 1

		return reward

	def all_delivered(self):
		for p in range(self.package_count):
			if not self.environment['packages']['delivered'][p]:
				return False
		return True

