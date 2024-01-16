from src.wien_env import WienEnv
from src.vis2 import Visualizer as Vis

environment_options = {
	'place_count': 30,
	'vehicle_count': 10,
	'package_count': 10,
}

env = WienEnv(**environment_options)

vis = Vis(environment_options)
#vis.test_colors()
vis.draw(env.get_info())
