# https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13
from typing import TypedDict

import numpy as np
import pandas as pd
import random
from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance




class WienGraph():

	def __init__(self, number_of_places=80, randomize_places=False):

		self.places = number_of_places
		# place indicies to pick from
		self.picks = random.sample(range(80), number_of_places) if randomize_places \
			else range(number_of_places)
		self.coordinates_file = '../data/places/places.csv'
		self.distances_file = '../data/travel_times/wien_travel_times.csv'

	def get_node_coordinates(self) -> np.array:
		distances = pd.read_csv(self.coordinates_file, sep=';')
		return distances.iloc[self.picks][['latitude', 'longitude']].values

	def get_edge_details(self) -> dict:

		edge_count = self.places * (self.places-1)
		print(f'\n# of places: {self.places}\ncalculated # of edges: {edge_count}')

		# get distance data
		distances = pd.read_csv(self.distances_file, sep=';', encoding='ISO-8859-1')
		distances = distances[distances['mode'] == 'bicycling']
		#print(distances)

		edge_lengths = np.empty(shape=edge_count)
		edge_endpoints = np.empty(shape=(edge_count, 2))

		for place1 in self.picks:
			for place2 in self.picks:
				if place1 == place2:
					continue

				# the distances table only has data for a single direction
				# being the smaller number to the larger
				# so when place1 is larger than place2, the table filter is swapped
				if place1 < place2:  # normal case
					distance = distances[
						(distances['place1index'] == place1) &
						(distances['place2index'] == place2)
					]
				else:
					distance = distances[
						(distances['place1index'] == place2) &
						(distances['place2index'] == place1)
					]

				distance = round(distance['duration'].values[0], 1)

				#print(f'\np1: {place1}\np2: {place2}\n\tdistance: {distance}')

				edge_lengths[place1*2], edge_lengths[place1*2 + 1] = 2 * [distance]

				edge_endpoints[place1*2] = [place1, place2]
				edge_endpoints[place1*2 + 1] = [place2, place1]

		return {
			'edge_lengths': edge_lengths,
			'edge_endpoints': edge_endpoints
		}


	def create_instance(self):

		edge_details = self.get_edge_details()

		return GraphInstance(
			nodes = self.get_node_coordinates(),
			edges = edge_details['edge_lengths'],
			edge_links = edge_details['edge_endpoints']
		)

	def debug(self):
		edge_details = self.get_edge_details()
		return (self.get_node_coordinates(), edge_details['edge_lengths'])



testnode, testedges = WienGraph(number_of_places=20).debug()