<h1 style="text-align: center;" align="center">Reinforcement Learning on the Vehicle Routing Problem</h1>
<p style="text-align: center;" align="center">
	194.077 Applied Deep Learning 2023W, VU, 2.0h, 3.0EC<br>
</p>

# Table of contents
* [Introduction](#introduction)
* [Setup](#setup)
* [Parameters](#parameters)
	* [Actions](#actions)
	* [Model options](#model_options)
	* [Environment options](#environment_options)
* [Usage](#usage)

## Introduction <a name="introduction"></a>

The efficient operation of a vehicle fleet in order to transfer goods has always been a difficult problem to tackle in the field of optimization. This problem has been formalized as the Vehicle Routing Problem (VRP), an extension of the Traveling Salesman Problem. The goal is to deliver as much as possible with the fewest resources. As one can imagine, there are virtually infinite possible variations to constrain the problem in, though they usually resemble those found in real-world freight companies. In my implementation, I attempt to solve a specific simplification where there are only bikes that are capable of handling 1 package. Furthermore, every bike automatically picks up &
delivers any packages it passes over. Having only 1 type of vehicle & 1 type of package massively simplifies the action space in the environment. The model does however have the ability to assign no action to a vehicle & have it wait in its current location.

## Setup <a name="setup"></a>

After cloning the repository,

1. First, open a terminal in the root directory of the project.

2. Next, to install the necessary dependencies, use `pip install -r requirements.txt` in a python version 3.11 environment.

3. Then navigate to the source directory with `cd src`

4. Lastly run `python main.py vis` to start a visualization with the default parameters.

## Parameters <a name="parameters"></a>

### Actions <a name='actions'></a>
 * '**train**' - run the model.learn() function & save the weights
 * '**test**' - use the model to run an episode
 * '**vis**' - display actions in the environment

### Model options <a name='model_options'></a>
 * **environment_count** - number of simultaneous environments to train/test on. In the visualization case, this argument is only used to load the correct model as only 1 environment can be visualized
 * **training_timesteps_k** - max number of iterations to train on multiplied by 1,000. This argument is used when training the model, otherwise it is only to load the correct model for test/vis

### Environment options <a name='environment_options'></a>
* **place_count** - number of places in the environment. This can be in the range from 1-80, & defaults to 80
* **vehicle_count** - number of vehicles, defaults to 10
* **package_count** - number of packages, defaults to 20
* **verbose** - print out vehicle & package info during each `env.step()`
* **verbose_trigger** - if verbose is False, this will activate verbosity anyway after this many steps. This is useful if the model gets stuck. Defaults to 100,000

## Usage <a name="usage"></a>

The command format is:

> `python main.py [action] [-environment_count] [-training_timesteps_k] [-place_count] [-vehicle_count] [-package_count] [-verbose] [-verbose_trigger]`

Training:
* For example, to train a model with 5 simultaneous environments for 2 million timesteps with an environment using 40 places, 12 vehicles, & 25 packages, the command would be `python main.py train 5 2000 40 12 25`

Testing/Visualization:
* To get a list of available models, use `ls models`
* The default model name format is "ppo_vrp_e`environment_count`-t`training_timesteps_k`_pvp-`place_count`-`vehicle_count`-`package_count`"
* To test a speicfic model, specify the parameters in the same way.
* For example, to test the model `ppo_vrp_e10-t100_pvp-80-10-20.zip`, simply run `python main.py test 10 100 80 10 20`