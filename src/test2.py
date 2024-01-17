from src.wien_env import WienEnv
from src.visualizer import Visualizer as Vis
import numpy as np

environment_options = {
	'place_count': 30,
	'vehicle_count': 10,
	'package_count': 10,
}

env = WienEnv(**environment_options)

vis = Vis(environment_options)
#vis.test_colors()
vis.draw(env.get_info())

done = False
while not done:
	obs, reward, done, _, info = env.step(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
	vis.draw(info)