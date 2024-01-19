<h1 style="text-align: center;" align="center">William Amminger</h1>
<p style="text-align: center;" align="center">
	194.077 Applied Deep Learning 2023W, VU, 2.0h, 3.0EC<br>
</p>

# Table of contents
1. [Introduction](#introduction)
2. [Usage](#usage)

## Introduction <a name="introduction"></a>

The efficient operation of a vehicle fleet in order to transfer goods has always been a difficult problem to tackle in the field of optimization. This problem has been formalized as the Vehicle Routing Problem (VRP), an extension of the Traveling Salesman Problem. The goal is to deliver as much as possible with the fewest resources. As one can imagine, there are virtually infinite possible variations to constrain the problem in, though they usually resemble those found in real-world freight companies. In my implementation, I attempt to solve a specific simplification where there are only bikes that are capable of handling 1 package. Furthermore, every bike automatically picks up &
delivers any packages it passes over. Having only 1 type of vehicle & 1 type of package massively simplifies the action space in the environment. The model does however have the ability to assign no action to a vehicle & have it wait in its current location.

## Usage <a name="usage"></a>

To install the necessary dependencies, use `pip install -r requirements.txt` in a python version 3.11 environment

