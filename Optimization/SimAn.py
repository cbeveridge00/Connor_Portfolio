import mlrose_hiive as mlrose
import numpy as np
import randomrule

#setup nap-sack problem
weights = [17,32,23,10,25,22,27,7,25,17,23,25,8,1,31,22,5,21,25,27,1,26,2,2,9,21,1,20,18,32,14,1,21,7,15,22,25,10,25,21,
           30,7,16,3,16,22,31,6,22,8]
values = [10,6,9,16,6,12,12,5,10,3,16,13,2,18,7,4,16,4,13,12,13,3,2,7,17,3,6,14,14,7,14,19,2,8,16,3,19,14,2,10,4,10,8,10,
          5,13,10,17,3,7]

fitness = mlrose.Knapsack(weights, values)



fitness2 = mlrose.CustomFitness(randomrule.randomRule_max)
problem = mlrose.DiscreteOpt(length = 400, fitness_fn = fitness2, maximize = True, max_val = 2)

best_state, best_fitness, _ = mlrose.simulated_annealing(problem, max_attempts=10, max_iters=1000,
                                                    init_state=None, random_state=799)
print('Random Rule Best:')
print(best_fitness)
print(best_state)


problemSak = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val = 2)
best_state, best_fitness, _ = mlrose.simulated_annealing(problemSak, max_attempts=10, max_iters=1000,
                                                        init_state=None, random_state=239)
print('Knapsac Best:')
print(best_fitness)
print(best_state)