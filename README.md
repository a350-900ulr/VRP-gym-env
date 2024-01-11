<h1 style="text-align: center;" align="center">William Amminger</h1>
<p style="text-align: center;" align="center">
	194.077 Applied Deep Learning 2023W, VU, 2.0h, 3.0EC<br>
	Assignment 2 - Hacking<br>
	Due 19th December 2023<br>
</p>

# Summary

The model was unfortunately unable to deliver a package, though it seems to be an error with the implementation as even after 30+ minutes of iterations, not a single package has been delivered by chance. Many issues were encountered in the project, especially the creation of a custom gymnasium environment. The code to create & run a model is found in main.py, though there is most likely much tweaking needed before it accomplishes anything.

## Changes from exercise 1

The original exercise 1 describe a multi-modal environment with different vehicle types with their own capabilities. This has been simplified into a single mode: a bicycle. Besides complexity reduction, this was also done to reduce the amount of distance data that would be required for multiple modes. The physical space has also been moved to Vienna from a random environment, since the problem is no longer representing containers shipping. The end result is a distance matrix of bicycle travel time between many real places in Vienna.

## * The error metric you specified + reward function

It was attempted to create an error metric of the amount of total time traveled by all vehicles. The reward function is the number of packages delivered + (initial package distance - current package distance) / 1000

## * The target of that error metric that you want to achieve + achieved reward

The goal was to get this number to be the same as the sum of distances each package would be from its target destination. This would be the ideal case where no bike travels without making a delivery. The maximum reward would be the number of packages + a small amount from the distances the packages traveled

## * The actually achieved value of that metric

The error is infinity, as the model never accomplishes the task in any given timeframe. The reward achieved was also 0

## * The amount of time you spent on each task, according to your own work breakdown

Acquiring data ~ 1 week of research & debugging to decide on proper data format
* one of the mistakes i realized i made was going into deep reinforcement learning with almost 0 knowledge of how it works
* it took a while deciding what framework to use before deciding on stable baselines 3

Building environment ~ 1 week. Many things took much longer than expected. I was hung up multiple times on strange errors.
* Spent a few days writing a Graph object for gymnasium spaces, but ultimately abandoned it due to errors. The code is still in /old/wien_graph.py
	* At first, i kept getting an error that the required type should be a Numpy array. This didn't really make that much sense, since the `Graph` space uses a `sample()` function when in the environment itself, which returns a `GraphInstance` object. 
	* When putting the required data into a custom-made `GraphInstance`, the error states that the observation must be a single value. I can only assume that this is because `GraphInstance` implements the class `NamedTuple`, which is essentially multiple values. Still, it does not make much sense that despite being a `gymnasium.space` object, it is not supported as an actual observation. 
	* Eventually, even after tweaking the observation space heavily, i ended up at the error stating the observation does not match the observation space:
		> ```The observation returned by the `reset()` method does not match the shape of the given observation space Graph(Box(16.2239, 48.274822, (20, 2), float32), Box(0.0, 33.0, (760, 3), float32)). Expected: None, actual shape: (2,)```
	* This error made even less sense, as the expected type is now "None" instead of a Numpy array. The returned shape (2,) is actually a nested Numpy array in attempts to match template ( (20, 2), (760, 3) ). 
	* I did not want to give up on all the work i had put into it, but encountering "Expected: None" made me feel like i was only getting deeper & deeper into a black hole & i should have simply started with a `Box` space instead. 
* Environment checker is very picky about the format of objects. I spent more time than expected getting everything to pass as a valid observation space. For example StableBaselines does not support a nested Dict, something i only found out after creating it. Furthermore, the errors were extremely hard to debug due to the complicated nature of spaces acting as a template while the instance values are held elsewhere.
* In the original phase to indicate an empty value, such as a vehicle having no package, I used "None". Later on, environment spaces required numerical arrays so it was changed to -1. At a later point I learned that the environment space MultiDiscrete actually does not handle negative numbers, requiring a restructuring of how the environment space was constructed. 

Defining action space ~ 1 day

Integrate into reinforcement learning model ~ 1 day, but I ran out of time

## Setup

To install the necessary dependencies, use `pip install -r requirements.txt` in a python version 3.12 environment
