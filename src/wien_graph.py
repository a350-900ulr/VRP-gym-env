# https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13

import numpy as np
import pandas as pd
import random
from gymnasium.spaces import GraphInstance, Graph, Box, Discrete
from typing import NamedTuple

class WienGraph(Graph):
	def __init__(self, number_of_places=80, randomize_places=False):
		"""
		Creates a :class:`GraphInstance` space instance defined by `gymnasium.spaces.Graph` using csv files of place coordinates & distances between them.

		:param number_of_places: How many places to keep from the full set of 80. This number is usually kept small as the number of edges grows exponentially.
		:param randomize_places: Whether to randomize the selected subset instead of keeping simple the first 20 or so. When using the full set, this parameter does nothing.
		"""

		self.template = {  # parameters for a gymnasium graph object
			'node_space': Box(low=16.22390075, high=48.2748237, shape=(20, 2)),
			'edge_space': Box(low=0, high=33, shape=(760, 3)),
		}

		super().__init__(*self.template.values())
		self.places = number_of_places

		# indices of places to pick from
		self.picks = random.sample(range(80), number_of_places) if randomize_places \
			else range(number_of_places)
		self.coordinates_file = '../data/places/places.csv'
		self.distances_file = '../data/travel_times/wien_travel_times.csv'

	def get_node_coordinates(self) -> np.array:
		"""
		Generates the 1st required object 'nodes'.
		:return: Distance table filtered by the chosen locations in `self.picks` & then by retaining only the longitude & latitude columns. Since there is no label column, the node is simply indicated by the row position.
		"""
		distances = pd.read_csv(self.coordinates_file, sep=';')
		return distances.iloc[self.picks][['latitude', 'longitude']].values

	def get_edge_details(self, verbose=False) -> dict:
		"""
		Creates the edge data, including the lengths in a (`self.places` * 1) np.array & the edge endpoints in a (`self.places` * 2) np.array.
		:param verbose: Prints initially calculated # of edges, then within the loop print the places, calculated indicies, & inserted distance value.
		:return: Dictionary of the 2nd & 3rd objects required for :class:`GraphInstance` with the keys 'edge_lengths' & 'edge_endpoints'.
		"""

		edge_count = self.places * (self.places-1) * 2

		if verbose:
			print(
				f'\n# of places: {self.places}\n'
				f'calculated # of edges: {edge_count} (both directions)'
			)

		# get distance data
		distances = pd.read_csv(self.distances_file, sep=';', encoding='ISO-8859-1')
		distances = distances[distances['mode'] == 'bicycling']

		edge_lengths = np.empty(shape=edge_count)
		edge_endpoints = np.empty(shape=(edge_count, 2))

		cali = 0  # calculated index
		for place1 in self.picks:
			for place2 in self.picks:
				if place1 == place2:
					continue

				# the distances table only has data for a single direction
				# being the smaller number to the larger
				# so when place1 is larger than place2, the table filter is swapped

				distance = distances[
					(distances['place1index'] == min(place1, place2)) &
					(distances['place2index'] == max(place1, place2))
				]

				distance = round(distance['duration'].values[0])

				if verbose:
					print(
						f"{place1:03d}_{place2:03d}->{cali:03d}+{cali+1:03d}"
						f"\n\tdistance: {distance}"
					)

				edge_lengths[cali], edge_lengths[cali + 1] = 2 * [distance]
				edge_endpoints[cali] = [place1, place2]
				edge_endpoints[cali + 1] = [place2, place1]

				cali += 2  # update calculated index

		return {
			'edge_lengths': edge_lengths,
			'edge_endpoints': edge_endpoints
		}

	def raw_output(self) -> tuple[np.array, np.array, np.array]:
		"""
		Intermediary function to combine the 3 required objects as a tuple, later to be converted into :class:`GraphInstance` via `create_instance`.
		:return: tuple of node coordinates, edge lengths, and edge endpoints.
		"""
		edge_details = self.get_edge_details()
		return (
			self.get_node_coordinates(),
			edge_details['edge_lengths'],
			edge_details['edge_endpoints']
		)

	def sample(self) -> GraphInstance:
		"""
		Converts raw output data into :class:`GraphInstance` object
		"""
		nodes, edges, edge_links = self.raw_output()
		return GraphInstance(nodes, edges, edge_links)

	def get_template(self) -> Graph:
		"""
		Returns `gymnasium.spaces` object as required in the observation space of the custom environment, being the bounding boxes of the nodes & the edge space
		"""
		return Graph(*self.template.values())


'''
testnode, testlen, testend = WienGraph(number_of_places=20).raw_output()

test = WienGraph().get_node_coordinates()


min(test[:, 1])
max(test[:, 0])
'''