# Optimization - Backpack problem

This is a project I built from scratch on local search in complex environments. Local search algorithms search from a start state to neighboring states. The goal here is optimization, aiming to find a best state according to some objective function. The objective function in this case is according to the backpack or knapsack problem: **"Given a set of different items, each one with an associated value and weight, determine which items you should pick in order to maximize the value of the items in the backpack without surpassing it's capacity.**

![backpack problem](https://github.com/cbeveridge00/Connor_Portfolio/blob/main/backpack/Optimization/knapsack.png?raw=true)

This includes the full solutions to the project. Full student README can be found in the project folder. 

### Educational Goals

This project aims to strengthen student skills in python and test algorithm understanding through implementation of the following algorithms:

 - Hill Climbing
 - Simulated Annealing
 - Genetic Algorithm

For hill climbing, students code the base algorithm in addition to code needed to get neighbors. For simulated annealing, students must utilize a decay function in addition to coding the algorithm and the successor function. For the genetic algorithm, students code all aspects needed including generating an initial population, reproduction, mutation, and "natural selection".


### Further Details


In addition to coding the algorithms, students incorporate them to produce a graph that compares the performance and behavior of each one. A set random seed is used to produce certain expected results:

![backpack graph](https://github.com/cbeveridge00/Connor_Portfolio/blob/main/backpack/Optimization/Knapsack_graph.png?raw=true)

Students also produce a write-up explaining the algorithms and their results on this problem including the results of in the graph. They should understand which type of problems these algorithms are suited for and that the genetic algorithm is superior for this problem.
