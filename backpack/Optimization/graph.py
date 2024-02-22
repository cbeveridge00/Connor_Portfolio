from matplotlib import pyplot
from hill import hill_climb
from annealing import sim_annealing
from genetic import genetic
import random
import numpy

random.seed(534)
_, fitness_curve_hc = hill_climb()
random.seed(534)
_, fitness_curve_sa = sim_annealing()
random.seed(534)
numpy.random.seed(534)
_, fitness_curve_ga = genetic()

y = 'Sim Annealing'
x = 'Hill Climb'
z = 'Genetic'
fig = pyplot.figure(figsize=(12,9))
pyplot.plot(range(len(fitness_curve_hc)), fitness_curve_hc,'b-o', label=x)
pyplot.plot(range(len(fitness_curve_sa[:800])), fitness_curve_sa[0:800],'y-3', label=y)
pyplot.plot(range(len(fitness_curve_ga)), fitness_curve_ga,'r-+', label=z)
pyplot.legend()
pyplot.xlim(0,800)
pyplot.ylim(0,230)
pyplot.xlabel('Iterations')
pyplot.title('Knapsack problem')
pyplot.ylabel('Fitness')
fig.savefig("Knapsack_graph.png")
