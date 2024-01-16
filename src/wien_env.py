# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
import random
from typing import Any
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict, MultiBinary, Box
from src.funcs import create_distance_matrix, filler, multi_disc
import numpy as np
import time


class WienEnv(gym.Env):
	def __init__(self,
		place_count: int = 80, vehicle_count: int = 10, package_count: int = 10,
		verbose: bool = False, verbose_trigger: int = 100_000
	):
		"""
		Custom Environment that follows gym interface. This Wien Environment creates a space for the vehicle routing problem within specific places in vienna. Observation space & action are explained further in the docstring above the respective variable in the __init__ function
		:param place_count: number of places to use, out of the 80 total

		"""
		assert 1 <= place_count <= 80, print('only 80 places available')

		super().__init__()

		self.place_count = place_count + 1
		self.vehicle_count = vehicle_count + 1
		self.package_count = package_count + 1
		self.clock = 0  # total time the episode has been running
		self.total_travel = 0  # sum of all distance traveled by all vehicles
		self.distance_matrix = create_distance_matrix(place_count)
		self.verbose = verbose
		self.verbose_trigger = verbose_trigger

		# initial values for environment itself. This is also used during self.step()
		self.environment = self.reset()[0]

		# all attributes below are used by the environment
		self.reward_range = (1, self.package_count)

		self.observation_space = Dict({
			# information of all location distances, regardless of how many locations end up
			# being used
			'distances': Box(
				low = 0,
				high = np.amax(self.distance_matrix),  # -> 91 in full matrix w/o buffer
				shape = (self.place_count, self.place_count),
				dtype = int
			),

			'v_id': MultiDiscrete(filler(self.vehicle_count, self.vehicle_count)),
			'v_available': MultiBinary(self.vehicle_count),
			'v_transit_start': MultiDiscrete(filler(self.vehicle_count, self.place_count)),
			'v_transit_end': MultiDiscrete(filler(self.vehicle_count, self.place_count)),
			'v_transit_remaining': MultiDiscrete(filler(
				self.vehicle_count, np.amax(self.distance_matrix)
			)),
			'v_has_package': MultiDiscrete(filler(self.vehicle_count, self.package_count)),

			'p_id': MultiDiscrete(filler(self.package_count, self.package_count)),
			'p_location_current': MultiDiscrete(filler(self.package_count, self.place_count)),
			'p_location_target':  MultiDiscrete(filler(self.package_count, self.place_count)),
			'p_carrying_vehicle': MultiDiscrete(filler(self.package_count, self.vehicle_count)),
			'p_delivered': MultiBinary(self.package_count),
		})
		'''
		distances:
			information of all location distances
			
		v_id:
			never actually used anywhere, but exists to help see how the code is structured. 
			Vehicles have an ID randing from [1, vehicle_count], as in other fields '0' 
			indicates no vehicle
		v_available:
			whether the vehicle is available for a dispatch decision (to be sent to the next
			location). This starts out as all true, & is set to false when a dispatch has been
			received. Set back to true when vehicle has reached a location
		v_transit_start:
			if the vehicle is in transit, this indicates the origin. If a vehicle is not moving,
			this simply indicates its position
		v_transit_end:
			if the vehicle is in transit, this indicates the target. If a vehicle is not moving,
			this also simply indicates its position. When trying to get the current position
			of a vehicle, this is used as it is more up to date than `v_transit_start`
		v_transit_remaining:
			integer indicating the remaining number of minutes before a vehicle reaches its
			destination. This is decremented by 1 every `step()` if not already 0.
		v_has_package:
			ID of package contained in vehicle. 0 if none
		
		p_id:
			same as v_id, only an indicator
		p_location_current:
			initialized as the origin location of a package. if this package is picked up by a
			vehicle, it remains as the vehicles starting location until the destination is
			reached. A package is delivered when this equals the target destination
		p_location_target:
			ID of the place that the package needs to be delivered to
		p_carrying_vehicle:
			ID of vehicle currently carrying package. 0 if none
		p_delivered:
			boolean to act as a shorthand for p_location_current == p_location_target
		'''

		# possible values are in the range of the number of locations
		self.action_space = MultiDiscrete(vehicle_count * [self.place_count])

	def step(self, action) -> tuple[dict, float, bool, bool, dict[str, Any]]:
		"""
		Run one timestep of the environment’s dynamics using the agent actions. First the vehicle location dispatch decisions are assigned to each vehicle where a dispatch is valid (the vehicle is available). When a vehicle is dispatched its remaining transit time is also calculated. Second each vehicle that is in transit is progressed by 1 unit.

		:param action: place for each vehicle to go to in the format MultiDiscrete(vehicle_count * [place_count])
		:return: observation, reward, terminated, truncated, info
		"""

		if not self.verbose and self.clock > self.verbose_trigger:
			self.verbose = True
			print(
				f'Clock ({self.clock} has exceeded limit {self.verbose_trigger} '
				f'set by verbose trigger. Verbosity is now enabled'
			)

		if self.verbose: print(f'{action}{"-"*32}')

		for dispatch_location in action:
			assert dispatch_location in range(self.place_count)

		self.clock += 1

		for v, vehicle_decision in enumerate(action, start=1):
			# helper functions to shorten navigation of the object environment dictionary
			def vehi(key): return self.environment[key][v]
			def vehi_set(key, val): self.environment[key][v] = val

			if vehi('v_transit_remaining') == 0:
				'''
				this if else block creates an intermediary pause between a vehicle reaching its
				destination & being dispatched to the next location, allowing
				`self.automate_packages()` enough time to make any vehicle deliver/pickup any
				packages before the vehicle leaves
				'''
				if vehi('v_available'):
					# dispatch to location 0 means no dispatch
					if vehicle_decision != 0:
						vehi_set('v_available', False)
						vehi_set('v_transit_end', vehicle_decision)
						vehi_set(
							'v_transit_remaining',
							self.distance_matrix[vehi('v_transit_start')][vehicle_decision]
						)
				else:
					vehi_set('v_transit_start', vehi('v_transit_end'))
					vehi_set('v_available', True)

		# progress each vehicle
		for v in range(self.vehicle_count):
			if self.environment['v_transit_remaining'][v] > 0:
				self.environment['v_transit_remaining'][v] -= 1
				self.total_travel += 1

		return (
			self.environment,
			self.automate_packages(),
			all(self.environment['p_delivered']),
			False,
			self.get_info()
		)

	def reset(self, seed=None, verbose=False) -> tuple:
		"""
		Sets the environment back to its original state. The environment object is re-initialized with random vehicle & package positions to be delivered.
		:param verbose: print out package location data
		:return: dict { environment, info }
		"""

		environment_object = {
			'distances': self.distance_matrix,

			'v_id': np.array(range(self.vehicle_count)),
			'v_available': filler(self.vehicle_count, True),
			'v_transit_start': filler(
				self.vehicle_count, self.place_count, True, True
			),
			'v_transit_end': filler(self.vehicle_count),
			'v_transit_remaining': filler(self.vehicle_count, 0),
			'v_has_package': filler(self.vehicle_count),

			'p_id': np.array(range(self.package_count)),
			'p_location_current': filler(
				self.package_count, self.place_count, True, True
			),
			# the target location will be generated later in the code to make sure it is not
			# the same as the starting location (location_current)
			'p_location_target': [0],
			'p_carrying_vehicle': filler(self.package_count),
			'p_delivered': filler(self.package_count, False)
		}

		if verbose:
			print(
				f"\ncurr: {len(environment_object['p_location_current'])}"
				f"\ntarg: {len(environment_object['p_location_target'])}"
			)

		# fill target destinations with something other than their starting location
		for p in range(1, self.package_count):
			valid_values = list(range(1, self.place_count))
			valid_values.remove(environment_object['p_location_current'][p])
			if verbose:
				print(f'\nindex: {p}')
			environment_object['p_location_target'].append(random.choice(valid_values))

		# set delivered status of 0 package to true
		environment_object['p_delivered'][0] = True

		return (
			environment_object,
			{
				self.clock,
				self.total_travel
			}
		)

	def automate_packages(self):
		"""
		automatically pick up packages when vehicle passes over & automatically drop off at target location
		:return: number of packages that have been delivered for reward function
		"""

		for p in range(1, self.package_count):

			# helper functions to shorten navigation of the object environment dictionary
			def pack(key): return self.environment[key][p]
			def pack_set(key, val): self.environment[key][p] = val

			# if a package is already delivered, there are no operations to be done
			if pack('p_delivered'): continue

			# if package is on a vehicle, make sure its location is updated
			if pack('p_carrying_vehicle') != 0:
				pack_set(
					'p_location_current',
					self.environment['v_transit_start'][pack('p_carrying_vehicle')]
				)

			# for any undelivered package currently not on a vehicle,
			# make sure any empty vehicles fulfill the request
			if pack('p_carrying_vehicle') == 0:
				for v in range(1, self.vehicle_count):
					def vehi(key): return self.environment[key][v]
					def vehi_set(key, val): self.environment[key][v] = val

					# if an empty vehicle passes by, pick up the package
					if (
						vehi('v_has_package') == 0 and
						vehi('v_transit_end') == pack('p_location_current') and
						vehi('v_transit_remaining') == 0  # vehicle is not moving
					):
						vehi_set('v_has_package', p)
						pack_set('p_carrying_vehicle', v)

			# if a package is not delivered but on a vehicle, check if it has been delivered
			else:

				# same helper functions as before, but now the index is its carrying vehicle
				def vehi(key):
					return self.environment[key][pack('p_carrying_vehicle')]
				def vehi_set(key, val):
					self.environment[key][pack('p_carrying_vehicle')] = val

				# check if package could be delivered
				if (
					vehi('v_transit_end') == pack('p_location_target') and
					vehi('v_transit_remaining') == 0
				):
					pack_set('p_carrying_vehicle', 0)
					pack_set('p_delivered', True)
					vehi_set('v_has_package', 0)

		return sum(self.environment['p_delivered'])

	def get_package_distances(self):
		"""
		Get the sum of all distances each package is away from its target. This is used to later calculate a micro-reward for the model getting packages closer to their destination.
		"""
		distance = 0
		for p, p_start in enumerate(self.environment['p_location_current']):
			distance += self.distance_matrix[p_start][self.environment['p_location_target'][p]]
		return distance

	def get_info(self,
		v_infos = (
			'v_available', 'v_transit_start', 'v_transit_end', 'v_transit_remaining',
			'v_has_package'
		),
		p_infos = (
			'p_location_current', 'p_location_target', 'p_carrying_vehicle', 'p_delivered'
		)
	) -> dict:
		"""
		:return: `self.environment` object but only keys shown in the default arguments for debugging & visualization
		"""

		if self.verbose:
			print('vehicle info:')
			for v_info in v_infos:
				print(f'\t{v_info}: {self.environment[v_info]}')
			print('package info:')
			for p_info in p_infos:
				print(f'\t{p_info}: {self.environment[p_info]}')
			time.sleep(.1)

		info_dict = {
			'time': self.clock,
			'total_travel': self.total_travel,
		}

		for key in v_infos + p_infos:
			info_dict[key] = self.environment[key]

		return info_dict


test = WienEnv(verbose=True)
test.reset()