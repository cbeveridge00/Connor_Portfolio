import mlrose_hiive as mlrose
import numpy as np
from matplotlib import pyplot
import timeit

fitness = mlrose.FlipFlop()
problem = mlrose.DiscreteOpt(length = 30, fitness_fn = fitness, maximize = True, max_val = 2)

best_statef = None
fitness_curve_rh = None
best_restart = 0
best_fitness_f = 0
'''
#do randomhill gridsearch
for j in [25,60,100,150,200,250,1000]:


    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                       max_attempts=60,
                                                                       restarts=j,
                                                                       curve=True,
                                                                       random_state=239)
    if best_fitness > best_fitness_f:
        best_fitness_f = best_fitness
        best_statef = best_state
        fitness_curve_rh = fitness_curve
        best_restart = j


print('Random Hill Best Best:')
print(best_fitness_f)
print(best_statef)
print(best_restart)
'''
best_fitness_sa = 0
best_state_sa = None
fitness_curve_sa = None
best_decay_sa = 0
best_temp = 0
'''
# do simulated annealing gridsearch
for i in [.99, .98, .96, .94, .93]:
    for j in [.1, .25, .5, 1, 2, 3, 5,10]:

        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
                                                                             schedule=mlrose.GeomDecay(init_temp=j,
                                                                                                       decay=i),
                                                                             max_iters=1000,
                                                                             max_attempts=15,
                                                                             curve=True, random_state=439)

        if best_fitness > best_fitness_sa:
            best_fitness_sa = best_fitness
            best_state_sa = best_state
            fitness_curve_sa = fitness_curve
            best_decay_sa = i
            best_temp = j

print('Sim Ann. Best:')
print(best_fitness_sa)
print(best_state_sa)
print(best_decay_sa)
print(best_temp)

# Do genetic Algorithm Gridsearch
best_fitness_ga = 0
best_state_ga = None
fitness_curve_ga = None
best_max_ga = None
best_pop = 0
best_mut = 0
best_t = 100000
pop = [20, 30, 40, 50, 75, 100, 150]
mut = [.4, .5, .6, .7, .8]
for j in pop:
    for k in mut:

        tic = timeit.default_timer()
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=j, mutation_prob=k,
                                                                     max_iters=1000, curve=True,
                                                                     random_state=723)
        toc = timeit.default_timer()
        tot = toc - tic

        if best_fitness == 29 and tot < best_t:
            best_fitness_ga = best_fitness
            best_state_ga = best_state
            fitness_curve_ga = fitness_curve
            best_pop = j
            best_mut = k
            best_t = tot

print('GA Best:')
print(best_fitness_ga)
print(best_state_ga)
print(best_pop)
print(best_mut)

# Do mimic gridsearch
best_fitness_mm = 0
best_state_mm = None
fitness_curve_mm = None
best_pop = 0
best_pct = 0
tot = 0
best_t = 100000
for j in [100,125,150]:
    for k in [.2, .3, .4, .5]:
        tic = timeit.default_timer()
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=j, keep_pct=k,
                                                               max_iters=1000, curve=True, random_state=781)

        toc = timeit.default_timer()
        tot = toc - tic
        print(best_fitness)
        if best_fitness == 29 and tot < best_t:
            best_fitness_mm = best_fitness
            best_state_mm = best_state
            fitness_curve_mm = fitness_curve
            best_pop = j
            best_pct = k
            best_t = tot
            break

print('Mimic Best:')
print(best_fitness_mm)
print(best_state_mm)
print(best_pop)
print(best_pct)

'''
#figure clock wall times
tot_rh = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                                restarts=100,
                                                                                init_state=None, curve=True,
                                                                                max_attempts=40,
                                                                                random_state=239)
    toc = timeit.default_timer()
    tot_rh += toc - tic

t_rh = tot_rh/5

tot_sa = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem,
                                                                              schedule = mlrose.GeomDecay(init_temp=.5),
                                                                              max_iters=1000,
                                                                              max_attempts=15,
                                                                              curve=True, random_state=439)
    toc = timeit.default_timer()
    tot_sa += toc - tic

t_sa = tot_sa/5

tot_ga = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=75, mutation_prob=.7,
                                                                          max_iters=1000, curve=True,
                                                                          random_state=723)
    toc = timeit.default_timer()
    tot_ga += toc - tic

t_ga = tot_ga/5

tot_mm = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state, best_fitness, fitness_curve_mm = mlrose.mimic(problem, pop_size=150, keep_pct=.3, max_attempts=4,
                                                              max_iters=1000, curve=True, random_state=781)
    toc = timeit.default_timer()
    tot_mm += toc - tic

t_mm = tot_mm/5

x = 'Sim Annealing (' + str(int(t_sa * 1000)) + ' ms)'
y = 'Random Hill (' + str(int(t_rh * 1000)) + ' ms)'
w = 'Genetic (' + str(int(t_ga * 1000)) + ' ms)'
z = 'Mimic (' + str(int(t_mm * 1000)) + ' ms)'
fig = pyplot.figure(figsize=(12,9))
pyplot.plot(range(len(fitness_curve_rh)), fitness_curve_rh,'b-o', label = y)
pyplot.plot(range(len(fitness_curve_sa)), fitness_curve_sa,'y-3', label = x)
pyplot.plot(range(len(fitness_curve_ga)), fitness_curve_ga,'r-+', label = w)
pyplot.plot(range(len(fitness_curve_mm)), fitness_curve_mm,'g-x', label = z)
pyplot.legend()
pyplot.xlim(0,125)
pyplot.ylim(15,30)
pyplot.xlabel('Iterations')
pyplot.title('Flip Flop')
pyplot.ylabel('Fitness (max 29)')
fig.savefig("flipflopcomp.png")