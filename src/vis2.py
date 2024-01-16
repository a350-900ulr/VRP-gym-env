import pygame as pg
import sys
import math
import pandas as pd
from pygame import Surface, Color
import numpy as np
import random
import colorsys


class Visualizer:
	def __init__(self, environment_arguments: dict, verbose: bool = False):
		self.verbose = verbose
		self.canvas_size = (1000, 1000)
		self.vienna_map = pg.image.load('../images/vienna_blank3_scaled_darkened.png')
		self.coordinates = pd.read_csv('../data/places/places.csv', sep=';')\
			.loc[:environment_arguments['place_count']]
		self.screen = pg.display.set_mode(self.canvas_size)

		self.colors = {
			'packages': self.generate_colors(environment_arguments['package_count']),
			'vehicles': self.generate_colors(environment_arguments['vehicle_count']),
		}


		pg.init()
		pg.display.set_caption("Bike Travel")

	def draw(self, env_info: dict):
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

		self.screen.fill(0)
		# draw image on the bottom of the canvas
		self.screen.blit(
			source = self.vienna_map,
			dest = (0, self.canvas_size[1] - self.vienna_map.get_size()[1])
		)

		self.draw_places()



		"""
		'v_available', 'v_transit_start', 'v_transit_end', 'v_transit_remaining', 'v_has_package'
		'p_location_current', 'p_location_target', 'p_carrying_vehicle', 'p_delivered'
		time, total travel
		"""
		self.draw_packages(env_info['p_location_current'])
		#print(f"circle: {(self.coordinates.loc[0, 'longitude'],self.coordinates.loc[0, 'latitude']),}")
		# pg.draw.circle(
		# 	self.screen,
		# 	color = (0, 255, 255),
		# 	center = self.convert_coordinates(
		# 		self.coordinates.loc[0, 'latitude'],
		# 		self.coordinates.loc[0, 'longitude']
		# 	),
		# 	radius = 100
		# )

		pg.display.flip()

	def convert_coordinates(self, latitude: float, longitude: float) -> [int, int]:
		return [
			np.interp(longitude, [16.212963, 16.5258723], [0, self.canvas_size[0]]),
			self.canvas_size[1] - np.interp(
				latitude, [48.141826, 48.27906], [0, self.vienna_map.get_size()[1]]
			),
		]

	def draw_places(self) -> None:


		for lat, lon in zip(self.coordinates['latitude'], self.coordinates['longitude']):
			# draw a red dot for each place
			pg.draw.circle(
				self.screen, (0, 0, 0),
				self.convert_coordinates(lat, lon), 5, 2,
			)

		#self.screen.blit(self.place_surface, (0, 0))


	def generate_colors(self, amount: int) -> list[Color]:
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
		test = self.generate_colors(30)

		for index, color in enumerate(test):
			pg.draw.circle(
				self.screen, color,
				[index * 30 + 30, 20], 10
			)
		pg.display.flip()

	def draw_packages(self, package_locations: list[int]) -> None:
		for index, location in enumerate(package_locations[1:]):
			position = self.convert_coordinates(
				self.coordinates.loc[location, 'latitude'],
				self.coordinates.loc[location, 'longitude']
			)
			if self.verbose:
				print(
					f'location: {location}'
					f'color: {self.colors["packages"][index]}'
					f"center: {position}"
				)
			pg.draw.circle(
				self.screen,
				color = self.colors['packages'][index],
				center = position,
				radius = 3,
			)

