# https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13

import gymnasium as gym
import numpy as np


from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance


class WienGraph(GraphInstance):


	def __init__(self):
		super().__init__()
		self.nodes = getCoordinates()


	"""A Graph space instance.

	* nodes (np.ndarray): an (n x ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
	* edges (Optional[np.ndarray]): an (m x ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
	* edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
	"""

	# [[x, y], [x, y], ...]
	nodes: nodes

	# [length, length, ...]
	edges: np.ndarray

	# [[node1, node2], [node1, node2] ...]
	edge_links: np.ndarray