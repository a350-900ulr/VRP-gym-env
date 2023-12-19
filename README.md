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

Building environment ~ 1 week. Many things took much longer than expected. I was hung up multiple times on strange errors. 
* Spent a few days writing a Graph object for gymnasium spaces, but ultimately abandoned it due to errors. The code is still in /old/wien_graph.py
* Environment checker is very picky about the format of objects. I spent more time than expected getting everything to pass as a valid observation space. Furthermore, the errors were extremely hard to debug due to the complicated nature of spaces acting as a template while the instance values are held elsewhere

Defining action space ~ 1 day

Integrate into reinforcement learning model ~ 1 day, ran out of time