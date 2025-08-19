# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
import random
from typing import Any
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict, MultiBinary, Box

from distances import create_distance_matrix
import numpy as np
import time


class ViennaEnv(gym.Env):
	def __init__(self,
		place_count: int = 80, vehicle_count: int = 10, package_count: int = 20,
		verbose: bool = False, verbose_trigger: int = 100_000
	):
		"""
		Custom Environment that follows gym interface. This Viena Environment creates a space
		for the vehicle routing problem within specific predefined places in vienna. Observation
		space & action are explained further in the docstring above the respective variable in
		the `__init__`.
		
		:param place_count: Number of places to use, out of the 80 total. Details about these
			places can be found in `data/places/places.csv`.
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
		
		self.package_ratios = [  # to update `self.observation_space['p_transit_extra']
			{
				'ideal': 0,  # holds the minimum required distance to deliver a package
				'actual': 1  # total distance each package has traveled on a vehicle,
				# padded by 1 to avoid division by 0
			} for _ in range(self.package_count)
		]
		
		# initial values for environment itself, this is also used during self.step()
		self.environment = self.reset()[0]
		
		self.observation_space = Dict({
			# # information of all location distances, regardless of how many locations end up
			# # being used. May not be needed since these values do not change.
			# 'distances': Box(
			# 	low=0,
			# 	high=np.amax(self.distance_matrix),  # 91 in full matrix w/o buffer
			# 	shape=(self.place_count, self.place_count),
			# 	dtype=int
			# ),

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
			'p_transit_extra': Box(low=0, high=1, shape=(self.package_count,))
		})
		'''
		v_id:
			Never actually used anywhere, but exists to help see how the code is structured.
			Vehicles have an ID randing from [1, vehicle_count], as in other fields '0' 
			indicates no vehicle.
		v_available:
			Whether the vehicle is available for a dispatch decision (to be sent to the next
			location). This starts out as all true, & is set to false when a dispatch has been
			received. Set back to true when vehicle has reached a location.
		v_transit_start:
			If the vehicle is in transit, this indicates the origin. If a vehicle is not moving,
			this simply indicates its position.
		v_transit_end:
			If the vehicle is in transit, this indicates the target. If a vehicle is not moving,
			this also simply indicates its position. When trying to get the current position
			of a vehicle, this is used as it is more up to date than `v_transit_start`.
		v_transit_remaining:
			Integer indicating the remaining number of minutes before a vehicle reaches its
			destination. This is decremented by 1 every `step()` if not already 0.
		v_has_package:
			ID of package contained in vehicle, 0 if none.
		
		p_id:
			Same as `v_id`, only an indicator.
		p_location_current:
			Initialized as the origin location of a package. If this package is picked up by a
			vehicle, it remains as the vehicles starting location until the destination is
			reached. A package is delivered when this equals the target destination.
		p_location_target:
			ID of the place that the package needs to be delivered to.
		p_carrying_vehicle:
			ID of vehicle currently carrying package, 0 if none.
		p_delivered:
			Boolean to act as a shorthand for `p_location_current == p_location_target`.
		p_transit_extra:
			The ratio between the ideal # of timesteps a package needs to get delivered
			(directly from origin to target) & the # of timesteps it actually took to get there.
			In the ideal case, this value would be 1, while taking double the time is 1/2. This
			is later used as a multiplier for the reward function.
		'''

		# possible values are in the range of the number of locations
		self.action_space = MultiDiscrete(vehicle_count * [self.place_count])
	
	def reset(self, seed=None, verbose=False) -> tuple[
		dict[str, Any],
		dict[str, int]
	]:
		"""
		Sets the environment back to its original state. The environment object is
		re-initialized with random vehicle & package positions to be delivered.
		
		:param seed: Alters `np_random_seed`.
		:param verbose: Print out package location data.
		:return: environment, info
		"""
		
		environment_object = {
			#'distances': self.distance_matrix,
			
			'v_id': np.array(range(self.vehicle_count)),
			'v_available': filler(self.vehicle_count, True),
			'v_transit_start': filler(
				self.vehicle_count, self.place_count, True, True
			),
			'v_transit_end': filler(self.vehicle_count),
			'v_transit_remaining': filler(self.vehicle_count, ),
			'v_has_package': filler(self.vehicle_count),
			
			'p_id': np.array(range(self.package_count)),
			'p_location_current': filler(
				self.package_count, self.place_count, True, True
			),
			# the target location will be generated later in the code to make sure it is not
			# the same as the starting location (`location_current`)
			'p_location_target': [0],
			'p_carrying_vehicle': filler(self.package_count),
			'p_delivered': filler(self.package_count, False),
			'p_transit_extra': filler(self.package_count,)
		}
		
		if verbose:
			print(
				f"\ncurr: {len(environment_object['p_location_current'])}"
				f"\ntarg: {len(environment_object['p_location_target'])}"
			)
		
		# fill target destinations with something other than their starting location
		for p in range(1, self.package_count):
			origin = environment_object['p_location_current'][p]
			
			valid_values = list(range(1, self.place_count))
			valid_values.remove(origin)
			if verbose:
				print(f'\nindex: {p}')
			
			target = random.choice(valid_values)
			environment_object['p_location_target'].append(target)
			
			# assign the ideal package distances
			self.package_ratios[p]['ideal'] = self.distance_matrix[origin][target] + 1
		
		# set the delivered status of package 0 to true, so that the simulation can still detect
		# when all packages are delivered
		environment_object['p_delivered'][0] = True
		
		return (
			environment_object,
			{
				'clock': self.clock,
				'total_travel': self.total_travel
			}
		)


	def step(self, action) -> tuple[dict, float, bool, bool, dict[str, Any]]:
		"""
		Run one timestep of the environment’s dynamics using the agent actions. 1st, the
		vehicle location dispatch decisions are assigned to each vehicle where a dispatch is
		valid (the vehicle is available). When a vehicle is dispatched, its remaining transit
		time is also calculated. 2nd, each vehicle in transit is progressed by 1 unit.

		:param action: Place for each vehicle to go to in the format
			`MultiDiscrete(vehicle_count * [place_count])`.
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

	def automate_packages(self):
		"""
		Automatically pick up / drop off packages when a vehicle passes over the correct
		location.
		
		:return: # of packages that have been delivered multiplied by their extra transit ratio
			for the reward function.
		"""

		for p in range(1, self.package_count):

			# helper functions to shorten navigation of the object environment dictionary
			def pack(key): return self.environment[key][p]
			def pack_set(key, val): self.environment[key][p] = val

			# if a package is already delivered, there are no operations to be done
			if pack('p_delivered'): continue

			# if a package is on a vehicle, make sure its location is up to date
			if (
				pack('p_carrying_vehicle') != 0 and
				(updated_location :=
					pack('p_location_current') !=
					self.environment['v_transit_start'][pack('p_carrying_vehicle')]
				)
			):
				pack_set('p_location_current', updated_location)
				
				# update the total travel time of a package
				self.package_ratios[p]['actual'] += 1

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
						if self.verbose: print(f'package {p} picked up by vehicle {v}')
						vehi_set('v_has_package', p)
						pack_set('p_carrying_vehicle', v)

			# if a package is not delivered but on a vehicle, check if it has been delivered
			else:
				# same helper functions as before, but now instead of checking every vehicle
				# only the index of its carrying vehicle is used
				def vehi(key):
					return self.environment[key][pack('p_carrying_vehicle')]
				def vehi_set(key, val):
					self.environment[key][pack('p_carrying_vehicle')] = val

				# check if package could be delivered
				if (
					vehi('v_transit_end') == pack('p_location_target') and
					vehi('v_transit_remaining') == 0
				):
					if self.verbose:
						print(f'package {p} delivered by vehicle {pack("p_carrying_vehicle")}')
					vehi_set('v_has_package', 0)
					pack_set('p_location_current', pack('p_location_target'))
					pack_set('p_carrying_vehicle', 0)
					pack_set('p_delivered', True)
					pack_set(
						'p_transit_extra',
						self.package_ratios[p]['ideal'] / self.package_ratios[p]['actual']
					)
		
		#return sum(self.environment['p_delivered']-1)
		return sum(self.environment['p_transit_extra'])

	def get_package_distances(self):
		"""
		Not used anywhere. This was an experiment to get the sum of all distances each package
		is away from its target to later calculate a micro-reward for the model getting packages
		closer to their destination.
		"""
		distance = 0
		for p, p_start in enumerate(self.environment['p_location_current']):
			distance += self.distance_matrix[p_start][self.environment['p_location_target'][p]]
		return distance

	def get_info(self,
		v_infos = (
			'v_transit_start', 'v_transit_end', 'v_transit_remaining', 'v_has_package'
		),
		p_infos = (
			'p_location_current',
			'p_location_target',
			'p_carrying_vehicle',
			'p_delivered'
		)
	) -> dict:
		"""
		:return: `self.environment` object but only keys shown in the default arguments for
			debugging & visualization
		"""

		if self.verbose:
			print('vehicle info:')
			for v_info in v_infos:
				print(f'\t{v_info}: {self.environment[v_info]}')
			print('package info:')
			for p_info in p_infos:
				print(f'\t{p_info}: {self.environment[p_info]}')
			time.sleep(.1)

		info_dict = {}

		for key in v_infos + p_infos:
			info_dict[key] = self.environment[key]
		
		'''
		At 1st every package ratio is above 1. However, this score is not relevant as the
		package has not moved yet. Thus the score reset to zero until it gets below 1. For
		display purposes, the score itself is multiplied by 100 & coerced to an integer.
		'''
		info_dict['p_ratios'] = []
		for p, steps in enumerate(self.package_ratios):
			
			ratio = steps['ideal'] / steps['actual']
			
			if not info_dict['p_delivered'][p] and ratio >= 1:
				info_dict['p_ratios'].append(0)
			else:
				info_dict['p_ratios'].append(int(ratio * 100))
		
		info_dict['time'] = self.clock
		info_dict['total_travel'] = self.total_travel
		
		return info_dict

def filler(
	amount: int, fill_with: Any = 0, random_int_up_to_fill: bool = False,
	zero_at_start = False
) -> np.array:
	"""
	:param amount: Length of list to return.
	:param fill_with: Value to fill the list with, or in the case of randomization fill with a
		random integer in range of `[0, fill_with)`
	:param random_int_up_to_fill: Choice to randomize an integer range.
	:param zero_at_start: Whether to replace the 1st value with a 0.
	:return: `np.array` of shape `(amount,)`.
	"""
	
	if random_int_up_to_fill:
		# starts at one for generating random place locations in the environment
		result = np.array([random.randrange(1, fill_with) for _ in range(amount)])
	else:
		result = np.array([fill_with] * amount)
	
	if zero_at_start:
		result[0] = 0
	
	return result


def multi_disc(amount: int, value_range: int, zero_as_none = False) -> MultiDiscrete:
	"""
	Shorthand for `MultiDiscrete(filler())` used by the observation space. This essentially
	creates a `MultiDiscrete` containing `amount` number of `Discrete` spaces with the range of
	`value_range`.
	
	:param amount: Number of `Discrete` objects in the return object
	:param value_range: The range of each `Discrete` object, indicating the possible range of
		values that a single Discrete element could be.
	:param zero_as_none: Whether to shift the range of each Discrete object by 1 so that 0
		represents the empty value. This is used by environment variables that could have a
		`None` value, such as when a vehicle contains no package.
	:return: A `MultiDiscrete` object of shape (amount, value_range)
	"""
	from gymnasium.spaces import MultiDiscrete
	if zero_as_none:
		return MultiDiscrete(filler(amount, value_range+1))
	else:
		return MultiDiscrete(filler(amount, value_range))

