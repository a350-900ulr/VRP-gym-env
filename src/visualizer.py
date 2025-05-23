import pygame as pg
import sys
import pandas as pd
from pygame import Color
import numpy as np
import random
import colorsys
from funcs import create_distance_matrix
import time
import warnings

class Visualizer:
	def __init__(self, environment_arguments: dict, verbose: bool = False):
		"""
		Provides a visualization of the WienEnv environment using PyGame.
		:param environment_arguments: dictionary containing the keywords argument used to create the environment
		:param verbose: print out package colors & locations
		"""
		if environment_arguments['vehicle_count'] > 20:
			warnings.warn('Visualizer is not optimized for more than 20 vehicles')
		if environment_arguments['package_count'] > 29:
			warnings.warn('Visualizer is not optimized for more than 29 packages')

		self.verbose = verbose
		self.canvas_size = (1000, 1000)
		self.vienna_map = pg.image.load('../images/vienna_blank3_scaled_darkened_more.png')
		self.place_circle = pg.image.load('../images/place_circle_60.png')
		self.place_circle_delivered = pg.image.load('../images/place_circle_delivered.png')
		self.distance_matrix = create_distance_matrix(environment_arguments['place_count'])
		self.coordinates = pd.read_csv('../data/places/places.csv', sep=';')\
			.loc[:environment_arguments['place_count']-1]
		self.screen = pg.display.set_mode(self.canvas_size)

		self.colors = {
			'vehicles': self.generate_colors(environment_arguments['vehicle_count']),
			'packages': self.generate_colors(environment_arguments['package_count']),
		}

		pg.init()
		self.screen.fill(0)
		pg.display.set_caption("Bike Travel")
		self.font = pg.font.SysFont('mono', 16)

	def draw(self, env_info: dict):
		"""
		main drawing function, called after every dispatch action
		:param env_info: the info object returned from the environment step function
		"""
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()



		# draw image on the bottom of the canvas
		self.screen.blit(
			source = self.vienna_map,
			dest = (0, self.canvas_size[1] - self.vienna_map.get_size()[1])
		)

		self.draw_info(env_info)
		self.draw_places()
		self.draw_packages(env_info)
		self.draw_vehicles(env_info)


		# update screen
		pg.display.flip()

	def draw_info(self, env_info: dict):
		"""
		draws info headers at the top of the screen from the `env.get_info()` object
		"""
		# display name for dictionary keys
		name = {
			'v_transit_start': 'Start',
			'v_transit_end': 'End',
			'v_transit_remaining': 'Remaining',
			'v_has_package': 'Package',

			'p_location_current': 'Location',
			'p_location_target': 'Target',
			'p_carrying_vehicle': 'Vehicle',
			'p_delivered': 'Delivered',

			'time':         'Game Clock       ',
			'total_travel': 'Total Travel Time',
		}
		text_offset_x = 0
		text_offset_y = 0
		for stat, values in env_info.items():
			# move environment details to the right side of the screen
			if stat == 'time':
				text_offset_x += self.canvas_size[0] / 1.4
				text_offset_y = 0

			# convert boolean list delivered status to integer
			if stat == 'p_delivered':
				values = list(map(int, values))

			# print vehicle header above the vehicle stats with their colors
			if stat == 'v_transit_start':
				self.text('Vehicles', text_offset_x + 15, text_offset_y * 32)
				for v, vehicle_color in enumerate(self.colors['vehicles']):
					self.draw_bike(
						140 + v * 30, text_offset_y * 30 + 20,
						vehicle_color
					)
				text_offset_y += 1

			# print package header above the package stats with their colors
			if stat == 'p_location_current':
				self.text('Packages', text_offset_x + 15, text_offset_y * 32)
				for p, package_color in enumerate(self.colors['packages']):
					self.draw_single_package(
						(140 + p * 30, text_offset_y * 30 + 30),
						package_color
					)
				text_offset_y += 1

			# finally, remove the 1st dummy value from stats & convert to padded string
			if stat not in ['time', 'total_travel']:
				values = self.pad_digits(values[1:])

			self.text(
				f'{name[stat]:<9}: {values}',
				text_offset_x + 15, text_offset_y * 32
			)
			text_offset_y += 1



	def pad_digits(self, input_list: list[int]) -> str:
		"""
		creates a string representation of an integer list, where any number less than 10 is padded with a space
		"""
		return ' '.join(f'{integer:2d}' for integer in input_list)

	def text(self, input_text: str, x: int = 32, y: int = 32):
		self.screen.blit(
			self.font.render(
				input_text,
				True,
				(255, 255, 255),
				(0, 0, 0),
			),
			(x, y+10)
		)
		pg.display.update()

	def convert_coordinates(self, latitude: float, longitude: float, offset = 0) -> list[float]:
		"""
		converts latitude and longitude to pixel coordinates. Ranges were made manually based on the corners of vienna_blank3_scaled_darkened.png
		:param latitude: between 48.141826, 48.27906
		:param longitude: between 16.212963, 16.5258723
		:param offset: this is used to center a shape when its position is specified by its top left corner
		:return: list of [x, y] pixel coordinates. Note that the values are somewhat flipped as latitude represents the y axis
		"""
		return [
			float(np.interp(
				longitude, [16.212963, 16.5258723], [0, self.canvas_size[0]]
			)) + offset,
			# subtract from canvas size to flip the y axis
			self.canvas_size[1] - float(np.interp(
				latitude, [48.141826, 48.27906], [0, self.vienna_map.get_size()[1]]
			)) + offset,
		]

	def get_position(self, location_index: int) -> tuple[float, float]:
		"""
		shorthand for calling `convert_coordinates` given only the index of a specific location
		:param location_index: cannot exceed the initial place count
		:return: tuple of x, y
		"""
		location_index -= 1
		list_object = self.convert_coordinates(
			self.coordinates.loc[location_index, 'latitude'],
			self.coordinates.loc[location_index, 'longitude']
		)
		return list_object[0], list_object[1]

	def draw_places(self):
		"""
		for every place chosen, draw a circle via the image specificed in `self.place_circle`
		"""
		for lat, lon in zip(self.coordinates['latitude'], self.coordinates['longitude']):
			self.screen.blit(
				source = self.place_circle,
				dest = (self.convert_coordinates(lat, lon, -5))
			)

	def generate_colors(self, amount: int) -> list[Color]:
		"""
		generate a list of random bright colors in the pygame.Color format
		:param amount: number of colors to generate
		:return: list of length `amount`
		"""
		# https://colorizer.org/
		output_list = []
		for _ in range(amount):
			# generate HLS values
			hue = random.randrange(0, 360)
			if 215 < hue < 265:  # force colors within blue/purple range to be brighter
				lightness = random.randrange(650, 800)
			else:
				lightness = random.randrange(500, 750)
			saturation = random.randrange(666, 1000)
			# convert to RGB
			rgb_color = colorsys.hls_to_rgb(hue / 360, lightness / 1000, saturation / 1000)
			# convert to list of Colors
			output_list.append(pg.Color(
				int(rgb_color[0] * 255),
				int(rgb_color[1] * 255),
				int(rgb_color[2] * 255))
			)
		return output_list

	def test_colors(self):
		"""
		debugging function used to show an example of the generated colors
		"""
		test = self.generate_colors(30)

		for index, color in enumerate(test):
			pg.draw.circle(
				self.screen, color,
				[index * 30 + 30, 20], 10
			)
		pg.display.flip()

	def draw_single_package(self, center, color):
		pg.draw.circle(
			self.screen,
			color = color,
			center = center,
			radius = 3,
		)

	def draw_packages(self, env: dict):
		"""
		if a package is not in transit, use the generated colors to draw a circle for each package inside the place circle
		:param env: info object from the environment step function to get the current package locations
		"""
		for index, location in enumerate(env['p_location_current'][1:], start=1):
			# if a package is on a vehicle (in transit), skip it
			if env['p_carrying_vehicle'][index] != 0:
				#print(f'package {index} is on a vehicle')
				continue

			position = self.get_position(location)

			if self.verbose:
				print(
					f'location: {location}'
					f'color: {self.colors["packages"][index-1]}'
					f"center: {position}"
				)
			#print(f'package {index} is not on a vehicle')
			self.draw_single_package(position, self.colors['packages'][index-1])

			if env['p_delivered'][index]:
				self.screen.blit(
					source = self.place_circle_delivered,
					dest = (position[0]-5, position[1]-5)
				)

	def draw_bike(self, x, y, vehicle_color):
		"""
		draws a bike at the specified position, where the x & y arguments indicate the middle point of the bike. Thus when this is draw on top of a location, it will be centered
		"""
		pg.draw.line(
			self.screen,
			color = vehicle_color,
			start_pos = (x-2, y-0),
			end_pos = (x+2, y+0),
		)
		pg.draw.circle(
			self.screen,
			color = vehicle_color,
			center = (x-3, y+1),
			radius = 2,
			width = 1
		)
		pg.draw.circle(
			self.screen,
			color = vehicle_color,
			center = (x+3, y+1),
			radius = 2,
			width = 1
		)

	def draw_vehicles(self, env: dict):
		# i am sorry
		for v in range(1, len(env['v_transit_start'])):
			def vehi(key): return env[key][v]

			if vehi('v_transit_remaining') == 0:  # bike is at a location
				# vehicle has not moved at all yet, thus it is the start of the simulation
				if vehi('v_transit_end') == 0:
					position = self.get_position(vehi('v_transit_start'))
				else:
					position = self.get_position(vehi('v_transit_end'))
			else:  # bike is in transit
				start = self.get_position(vehi('v_transit_start'))
				end = self.get_position(vehi('v_transit_end'))
				distance = self.distance_matrix[vehi('v_transit_start')][vehi('v_transit_end')]

				position = (
					float(np.interp(
						distance - vehi('v_transit_remaining'),
						[0, distance], [start[0], end[0]]
					)),
					float(np.interp(
						distance - vehi('v_transit_remaining'),
						[0, distance], [start[1], end[1]]
					)),
				)

			self.draw_bike(
				position[0], position[1],
				self.colors['vehicles'][v-1]
			)

			if vehi('v_has_package') != 0:
				pg.draw.rect(
					self.screen,
					color = self.colors['packages'][vehi('v_has_package')-1],
					rect = (position[0]-2, position[1]-3, 4, 3),
				)

# test = {
# 	'v_transit_start':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
# 	'v_transit_end':       [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 54, 44, 33, 22, 13],
# 	'v_transit_remaining': [0, 7, 5, 6, 4, 2, 6, 1, 10, 23, 32, 23, 1, 1, 2],
# 	'v_has_package':       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
# 	'p_location_current':  [0, 1, 2],
# 	'p_location_target': [0, 3, 6],
# 	'p_carrying_vehicle': [0, 0, 0],
# 	'p_delivered': [0, 0, 0],
#
# 	'time': 100,
# 	'total_travel': 500,
# }
#
# environment_options = {
# 	'place_count': 80,
# 	'vehicle_count': len(test['v_transit_start'])-1,
# 	'package_count': len(test['p_location_current'])-1
# }
#
# vis = Visualizer(environment_options)
# vis.draw(test)
# time.sleep(100)