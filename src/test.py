
import numpy as np
import pandas as pd

def create_distance_matrix():
	distances = pd.read_csv(
		'../data/travel_times/wien_travel_times.csv',
		sep = ';',
		encoding = "ISO-8859-1"
	)

	distances = distances[distances['mode'] == 'bicycling']

	del distances['mode']

	dist_matrix = np.zeros((80, 80))

	for place1 in range(80):
		for place2 in range(place1+1, 80):
			# set both values on both sides of the diagonal
			dist_matrix[place1, place2], dist_matrix[place2, place1] = 2 * [round(
				distances[
					(distances['place1index'] == place1) &
					(distances['place2index'] == place2)
					]['duration'].values[0],
				1)]  # round to 1 decimal place ~ 6 seconds

	return dist_matrix





"""A Graph space instance.

* nodes (np.ndarray): an (n x ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
* edges (Optional[np.ndarray]): an (m x ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
* edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
"""















































# https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13

from typing import NamedTuple
import gymnasium as gym
import numpy as np

from wien_graph import WienGraph
from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance



class testClass(NamedTuple):
	bepis: list

tester = testClass([x for x in range(10)])

tester.bepis