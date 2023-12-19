<h1 style="text-align: center;" align="center">William Amminger</h1>
<p style="text-align: center;" align="center">
	194.077 Applied Deep Learning 2023W, VU, 2.0h, 3.0EC<br>
	Assignment 2 - Hacking<br>
	Due 19th December 2023<br>
</p>

# Summary of results

First of all, no results were achieved. The model was unable to deliver a package, though it seems to be an error with the implementation as even after 30+ minutes of iterations, not a single package has been delivered by chance. Many issues were encountered in the project, especially the creation of a custom gymnasium environment. The code to create & run a model is found in main.py, though there is most likely much tweaking needed before it accomplishes anything.

## Changes from exercise 1

The original exercise 1 describe a multi-modal environment with different vehicle types with their own capabilities. This has been simplified into a single mode: a bicycle. Besides complexity reduction, this was also done to reduce the amount of distance data that would be required for multiple modes. The physical space has also been moved to Vienna from a random environment, since the problem is no longer representing containers shipping. The end result is a distance matrix of bicycle travel time between many real places in Vienna. 


## * The error metric you specified

It was attempted to create an error metric of the amount of total time traveled by all vehicles. 

## * The target of that error metric that you want to achieve

The goal was to get this number to be the same as the sum of distances each package would be from its target destination. This would be the ideal case where no bike travels without making a delivery.

## * The actually achieved value of that metric

The error is infinity, as the model never accomplishes the task in any given timeframe.

## * The amount of time you spent on each task, according to your own work breakdown

> Acquiring data ~ 1 week of research & debugging to decide on proper data format

> Building environment ~ 1 week. Many things took much longer than expected. I was hung up multiple times on strange errors. I spent a few days writing a Graph object for gymnasium spaces, but ultimate abandoned it due to errors. The code is still in /old/wien_graph.py

> Defining action space ~ 1 day

> Integrate into reinforcement learning model ~ 1 day, ran out of time