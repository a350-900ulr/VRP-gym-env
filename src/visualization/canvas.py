import pygame as pg
import sys
import math
import pandas as pd


coordinates = pd.read_csv('../../data/places/places.csv', sep=';')
print(min(coordinates['latitude']))
print(max(coordinates['latitude']))
print(min(coordinates['longitude']))
print(max(coordinates['longitude']))



# Initialize Pygame
pg.init()

canvas_size = (1280, 720)

screen = pg.display.set_mode(canvas_size)
pg.display.set_caption("Bike Travel")

vienna_map = pg.transform.scale(pg.image.load('images/vienna_map_image.png'), (720, 720))

# Get the dimensions of the image
image_width, image_height = vienna_map.get_size()

# Calculate the position to display the image on the right side
image_x = canvas_size[0] - image_width
image_y = (canvas_size[1] - image_height) // 2

# Main game loop
while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit()

	# Clear the screen
	screen.fill((0, 0, 0))

	# Draw the image on the right side
	screen.blit(vienna_map, (image_x, image_y))

	# Update the display
	pg.display.flip()
