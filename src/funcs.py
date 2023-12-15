

def create_distance_matrix():
	import pandas as pd
	import numpy as np

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
				]['duration'].values[0]
			)]

	return dist_matrix


def filler(amount: int, fill_with=None, random_int_up_to_fill: bool=False):
	"""
	:param amount: length of list to return
	:param fill_with: value to fill the list with, or in the case of randomization fill with a andom integer in range of [0, fill_with)
	:param random_int_up_to_fill: choice to randomize an integer range
	:return: list filled with specified values
	"""
	import random
	if random_int_up_to_fill:
		return [random.randrange(fill_with) for _ in range(amount)]
	else:
		return [fill_with for _ in range(amount)]