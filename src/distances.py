import pandas as pd
import numpy as np

def create_distance_matrix(env_places: int = 20, buffer: int = 2) -> np.array:
	"""
	Uses the travel times file to generate a matrix of the distances between them, rounded to
	the nearest integer + 2.
	
	:param env_places: Specified number of places in the environment.
	:param buffer: Increase all values by a certain amount before assigning to matrix. The
		default value is 2 minutes as an approximate time it takes to pickup/unload a package.
	:return: Numpy array of shape `(env_places, env_places)`
	"""

	distances = pd.read_csv(
		'../data/travel_times/wien_travel_times.csv',
		sep = ';',
		encoding = "ISO-8859-1"
	)

	distances = distances[distances['mode'] == 'bicycling']

	"""
	Dimensions have +1 appended to have the 0th row/column be blank. This is to keep in line 
	with the environment, where dispatching to place 0 indicates the vehicle should not move yet.
	"""
	dist_matrix = np.zeros((env_places+1, env_places+1), dtype=int)

	for place1 in range(env_places):
		for place2 in range(place1+1, env_places):
			"""
			This sets the values on both sides of the diagonal, thus it is mirrored.
			The indices here are also offset by +1 as explained in the previous block comment.
			"""
			dist_matrix[place1+1, place2+1], dist_matrix[place2+1, place1+1] = 2 * [round(
				distances[
					(distances['place1index'] == place1) &
					(distances['place2index'] == place2)
				]['duration'].values[0]
			) + buffer]  # artificially increase all transit times to simulate loading times

	return dist_matrix


