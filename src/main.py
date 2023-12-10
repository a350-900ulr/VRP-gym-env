places = 10  # locations to use, from 0-80
packages = 200
vehicles = 20

'''
https://keras.io/examples/rl/ppo_cartpole/
https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl#50663200

ideas: limit time riding (as episode), make part of reward distance packet is from target
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time

# distance matrix


observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
	mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)
















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


test = create_distance_matrix()


