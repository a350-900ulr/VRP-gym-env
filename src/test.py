# https://github.com/openai/gym/blob/master/gym/spaces/graph.py#L13

from typing import NamedTuple
import gymnasium as gym
import numpy as np


from gymnasium.spaces import Tuple, MultiDiscrete, Dict, Graph, Discrete, GraphInstance



class testClass(NamedTuple):
	bepis: list

tester = testClass([x for x in range(10)])

tester.bepis