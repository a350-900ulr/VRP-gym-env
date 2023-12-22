'''
file to test random stuff


links:
https://chat.openai.com
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
https://gymnasium.farama.org/api/env/
https://www.gymlibrary.dev/api/spaces/
https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13
'''


def create_distance_matrix():
	distances = pd.read_csv(
		'../../data/travel_times/wien_travel_times.csv',
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

# how
test = create_distance_matrix()

test[1][2]



import pandas as pd
import matplotlib.pyplot as plt

distances = pd.read_csv(
	'../../data/travel_times/wien_travel_times.csv',
	sep = ';',
	encoding = "ISO-8859-1"
)

distances = distances[distances['mode'] == 'bicycling']

del distances['mode']
plt.hist(distances['duration'], bins=50)




















# https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13

from typing import NamedTuple


class testClass(NamedTuple):
	bepis: list

tester = testClass([x for x in range(10)])

tester.bepis












picks = range(20)
cali = 0
for place1 in range(len(picks)):
	for place2 in range(place1+1, len(picks)):


		#cali = (place1)*2 + place1*len(picks) + (place2)*2  # calculate index

		print(f"{place1:03d}_{place2:03d}->{cali:03d}+{cali+1:03d}")
		cali += 2













import src.funcs as fn
import numpy as np
test = fn.create_distance_matrix()
np.amax(test)

import numpy as np
from src.old.wien_graph import WienGraph
test = np.array([*WienGraph().raw_output()], dtype=object)


testnode, testlen, testend = WienGraph(number_of_places=20).raw_output()
test = np.c_[ testend, testlen ]


arr1 = np.arange(4)
arr2 = np.arange(5,7)
arr3 = np.arange(7,12)

array_of_arrays = np.array([arr1, arr2, arr3])
array_of_arrays
np.concatenate(array_of_arrays)





from stable_baselines3.common.env_checker import check_env
#from src.wien_env import WienEnv
from src.old.wien_env_graphtest import WienEnvGraph

check_env(WienEnvGraph())










