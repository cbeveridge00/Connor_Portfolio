import mlrose_hiive as mlrose
import numpy as np
import randomrule
from matplotlib import pyplot
#


#setup nap-sack problem
weights = [17,32,23,10,25,22,27,7,25,17,23,25,8,1,31,22,5,21,25,27,1,26,2,2,9,21,1,20,18,32,14,1,21,7,15,22,25,10,25,21,
           30,7,16,3,16,22,31,6,22,8]
values = [10,6,9,16,6,12,12,5,10,3,16,13,2,18,7,4,16,4,13,12,13,3,2,7,17,3,6,14,14,7,14,19,2,8,16,3,19,14,2,10,4,10,8,10,
          5,13,10,17,3,7]

fitness = mlrose.Knapsack(weights, values)



fitness2 = mlrose.CustomFitness(randomrule.randomRule_max)
problem = mlrose.DiscreteOpt(length = 400, fitness_fn = fitness2, maximize = True, max_val = 2)

best_state, best_fitness, fitness_curve1 = mlrose.random_hill_climb(problem, max_attempts=200, max_iters=1000, restarts=15,
                                                    init_state=None, curve=True, random_state=799)
print('Random Rule Best:')
print(best_fitness)
print(best_state)


problemSak = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val = 2)
best_state, best_fitness, fitness_curve2 = mlrose.random_hill_climb(problemSak, max_attempts=10, max_iters=1000, restarts=10,
                                                        init_state=None, curve=False, random_state=239)
print('Knapsac Best:')
print(best_fitness)
print(best_state)

fig = pyplot.figure(figsize=(12,9))
pyplot.plot(range(1000), fitness_curve1,'g-o')
pyplot.xlim(50,1000)
pyplot.ylim(0,400)
pyplot.xlabel('Iterations')
pyplot.title('Random Rule')
pyplot.ylabel('Fitness')
fig.savefig("a.png")