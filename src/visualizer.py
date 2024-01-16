import pygame as pg
import sys
import math
import pandas as pd
from pygame import Surface
import numpy as np


class Visualizer:
	def __init__(self, place_count: int = 80, canvas_size: tuple = (1000, 1000)):
		self.place_count = place_count
		self.canvas_size = canvas_size
		self.vienna_map = pg.image.load('../images/vienna_blank3_scaled.png')
		self.coordinates = pd.read_csv('../data/places/places.csv', sep=';').loc[:place_count]
		self.screen = pg.display.set_mode(self.canvas_size)

		pg.init()
		pg.display.set_caption("Bike Travel")

	def draw(self):
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

		self.screen.fill(0)
		# shift image to bottom of canvas
		self.screen.blit(self.vienna_map, (0, self.canvas_size[1] - self.vienna_map.get_size()[1]))
		self.draw_place_indicators()




		pg.display.flip()

	def convert_coordinates(self, latitude: float, longitude: float) -> [int, int]:
		return [
			np.interp(longitude, [16.212963, 16.5258723], [0, self.canvas_size[0]]),
			self.canvas_size[1] - np.interp(
				latitude, [48.141826, 48.27906], [0, self.vienna_map.get_size()[1]]
			),
		]

	def draw_place_indicators(self) -> None:
		for lat, lon in zip(self.coordinates['latitude'], self.coordinates['longitude']):
			# draw a red dot for each place
			pg.draw.circle(
				self.screen, (255, 0, 0),
				self.convert_coordinates(lat, lon), 5, 2,
			)


test  = Visualizer().draw()
