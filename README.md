<h1 style="text-align: center;">William Amminger</h1>
<p style="text-align: center;">
	194.077 Applied Deep Learning 2023W, VU, 2.0h, 3.0EC<br>
	Assignment 1 - Initiate<br>
	Due 24th October 2023<br>
</p>

### 1. References to at least two scientific papers that are related to your topic

> [Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem](https://arxiv.org/abs/2110.02629)

> [DeepFreight: Integrating Deep Reinforcement Learning and Mixed Integer Programming for Multi-transfer Truck Freight Delivery ](https://paperswithcode.com/paper/deepfreight-a-model-free-deep-reinforcement)

### 2. A decision of a topic of your choice

> ****Implementation of Reinforcement Learning into the Vehicle Routing Problem (VRP), an extension of the Traveling Salesman Problem. This involves a fully connected graph of geographical points that must be visited by a fleet of vehicles in order to fulfill a set of deliveries. The optimization function is to minimize the number of vehicles used & the total fuel used (distance traveled) by all vehicles. Common constraints to the function typically resemble those found in real-world freight companies. Examples include the maximum capacity of a vehicle, the maximum distance it is allowed to travel at once, & priority of a parcel. Additionally, there are many variants of the problem which add additional parameters into the environment, such as a specific time window a parcel needs to be delivered, whether parcels can be transferred intermittedly between vehicles, & unloading times for dropping off a specific parcel based on its position in the vehicle.****

### 3. A decision of which type of project you want to do

> This would be mainly a "Bring your own data" type of project. The dataset will be  a mix of generated geolocation data from the Google Maps API along with some randomized travel times & locations in aspects where real world data is not easy to automatically acquire.

### 4. A written summary that should contain:

#### a. Short description of your project idea and the approach you intend to use

> This project will be a simple variant of the VRP with parcels that exists at various points in an undirected graph, having an origin & destination. A fleet of vehicles will find a way to achieve all deliveries using the least of amount of resources possible.

> In this case, I would like to attempt creating a multi-modal system where there are many types of vehicles with different capabilities that are limited to paths which have the corresponding infrastructure for the vehicle. This is similar to real world container shipping, where modes of transportation requires the consideration of each mode's abilities. Thus, to keep modes competitive the amount of time required for parcels to be delivered will also be part of the cost function. The cost function can at first be a simple sum of (fuel used + time needed), with modifications later on to determine the effects.

> For initial simplicity, a parcel will not have a weight, priority, or profit. The resulting setup also includes features from common variants of VRP, including
> * **Capacitated Vehicles** that have a limited amount carrying space for parcels
> * **Open Vehicle Routing** so that a vehicle does not have to return to a specific point
> * **Pickup and Delivery** for each parcel, meaning that their origin & destination can be any location, or city in this implementation.

> To put this in a reinforcement learning structure, the basic components are defined as
> * **Agent**: Shipping company that can allocate routes to vehicles
> * **Environment**: Locations scattered geographically, with infrastructure existing between them for a set amount of vehicles
> * **Action**: Set a target route for each vehicle
> * **State**: The position of all vehicles
> * **Reward**: Amount of parcels delivered

#### b. Description of the dataset you are about to use (or collect)

> The dataset will contain objects with their respective attributes described below:
> * **Vehicle**: Capacity, Speed, Consumption, Max Distance
> * **Depot**: Location
> * **Parcel**: Origin, Destination, Size
> * **Edge**: Distance (for each mode)

#### c. A work-breakdown structure for the individual tasks with time estimates (hours or days) for dataset collection; designing and building an appropriate network; training and fine-tuning that network; building an application to present the results; writing the final report; preparing the presentation of your work.

Due 19th December 2023


he error metric you specified
the target of that error metric that you want to achieve
the actually achieved value of that metric
the amount of time you spent on each task, according to your own work breakdown structure.



Due 16th January 2024

    What is the problem that you tried to solve?
    Why is it a problem?
    What is your solution?
    Why is it a solution? (And in particular, why is or isn’t deep learning a solution?)
    
    final report, presentaiton