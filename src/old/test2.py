from src.wien_env import WienEnv
from src.visualizer import Visualizer as Vis
import numpy as np

environment_options = {
	'place_count': 6,
	'vehicle_count': 1,
	'package_count': 1,
}

env = WienEnv(**environment_options)

vis = Vis(environment_options)
#vis.test_colors()
vis.draw(env.get_info())

done = False
while not done:
	print('.', end='')
	obs, reward, done, _, info = env.step(np.array([x+1 for x in range(environment_options['vehicle_count'])]))
	vis.draw(info)










from src.visualizer import Visualizer as Vis

test = {
	'v_transit_start': [0, 2],
	'v_transit_end': [0, 3],
	'v_transit_remaining': [0, 7],
	'v_has_package': [0, 0],

	'p_location_current': [0, 1, 2],
	'p_location_target': [0, 3, 6],
	'p_carrying_vehicle': [0, 0, 0]
}

environment_options = {
	'place_count': 6,
	'vehicle_count': 1,
	'package_count': 2,
}

vis = Vis(environment_options)
vis.draw(test)


























"""Draw text to the screen."""
import pygame
from pygame.locals import *
import time

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

pygame.init()
screen = pygame.display.set_mode((640, 240))

sysfont = pygame.font.get_default_font()
print('system font :', sysfont)

t0 = time.time()
font = pygame.font.SysFont(None, 48)
print('time needed for Font creation :', time.time()-t0)

img = font.render(sysfont, True, RED)
rect = img.get_rect()
pygame.draw.rect(img, BLUE, rect, 1)

font1 = pygame.font.SysFont('chalkduster.ttf', 72)
img1 = font1.render('chalkduster.ttf', True, BLUE)

font2 = pygame.font.SysFont('didot.ttc', 72)
img2 = font2.render('didot.ttc', True, GREEN)

fonts = pygame.font.get_fonts()
print(len(fonts))
for i in range(7):
	print(fonts[i])

running = True
background = GRAY
while running:
	for event in pygame.event.get():
		if event.type == QUIT:
			running = False

	screen.fill(background)
	screen.blit(img, (20, 20))
	screen.blit(img1, (20, 50))
	screen.blit(img2, (20, 120))
	pygame.display.update()

pygame.quit()