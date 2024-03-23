import mlrose_hiive as mlrose
import numpy as np
from matplotlib import pyplot
import timeit
#


#setup nap-sack problem
weights = [17,32,23,10,25,22,27,7,25,17,23,25,8,1,31,22,5,21,25,27,1,26,2,2,9,21,1,20,18,32,14,1,21,7,15,22,25,10,25,21,
           30,7,16,3,16,22,31,6,22,8]
values = [10,6,9,16,6,12,12,5,10,3,16,13,2,18,7,4,16,4,13,12,13,3,2,7,17,3,6,14,14,7,14,19,2,8,16,3,19,14,2,10,4,10,8,10,
          5,13,10,17,3,7]

fitness = mlrose.Knapsack(weights[:30], values[:30])
problemSak = mlrose.DiscreteOpt(length=30, fitness_fn=fitness, maximize=True, max_val=2)
'''
best_statef = None
fitness_curve_rh = None
best_max_at = 0
best_restart = 0
best_fitness_f = 0

#do randomhill gridsearch
for j in [500,1000, 1500,2000]:
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problemSak, max_iters=1000,
                                                                        restarts=j,
                                                                        init_state=np.array([0] * 30), curve=True,
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

best_fitness_sa = 0
best_state_sa = None
fitness_curve_sa = None
best_max_sa = None
best_temp = 0
#do simulated annealing gridsearch
for i in [.9995,.999,.998, .995, .99, .98]:
    for j in [1,5,10,15,20]:

        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problemSak, schedule = mlrose.GeomDecay(init_temp=j, decay= i),
                                                                             max_attempts=50, max_iters=1000,
                                                                init_state=np.array([0] * 30), curve=True, random_state=439)

        if best_fitness > best_fitness_sa:
            best_fitness_sa = best_fitness
            best_state_sa = best_state
            fitness_curve_sa = fitness_curve
            best_max_sa = i
            best_temp = j

print('Sim Ann. Best:')
print(best_fitness_sa)
print(best_state_sa)
print(best_max_sa)
print(best_temp)

# Do genetic Algorithm Gridsearch
best_fitness_ga = 0
best_state_ga = None
fitness_curve_ga = None
best_pop = 0
best_mut = 0

for j in [25,50,100,250]:
    for k in [.05, .1, .25, .5]:

        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problemSak, pop_size=j, mutation_prob=k,
                                                                     max_attempts=15, max_iters=1000, curve=True,
                                                                     random_state=723)
        if best_fitness > best_fitness_ga:
            best_fitness_ga = best_fitness
            best_state_ga = best_state
            fitness_curve_ga = fitness_curve
            best_pop = j
            best_mut = k

print('GA Best:')
print(best_fitness_ga)
print(best_state_ga)
print(best_pop)
print(best_mut)





#Do mimic gridsearch
best_fitness_mm = 0
best_state_mm = None
fitness_curve_mm = None
best_pop = 0
best_pct = 0

for j in [500,600,750,900]:
    for k in [.05, .1, .15]:

        best_state, best_fitness, fitness_curve = mlrose.mimic(problemSak, pop_size=j, keep_pct=k, max_attempts=5,
                                                        max_iters=1000, curve=True, random_state=781)
        if best_fitness > best_fitness_mm:
            best_fitness_mm = best_fitness
            best_state_mm = best_state
            fitness_curve_mm = fitness_curve
            best_pop = j
            best_pct = k

print('MiMMic Best:')
print(best_fitness_mm)
print(best_state_mm)
print(best_pop)
print(best_pct)



'''








#figure clock wall times
tot_rh = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problemSak, max_iters=1000,
                                                                                restarts=1000,
                                                                                init_state=np.array([0] * 30),
                                                                                curve=True,
                                                                                random_state=239)
    toc = timeit.default_timer()
    tot_rh += toc - tic

t_rh = tot_rh/5

tot_sa = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problemSak, schedule=mlrose.GeomDecay(
                                                                                  init_temp=10, decay=.998),
                                                                                  max_attempts=50, max_iters=1000,
                                                                                  init_state=np.array([0] * 30),
                                                                                  curve=True, random_state=439)
    toc = timeit.default_timer()
    tot_sa += toc - tic

t_sa = tot_sa/5

tot_ga = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problemSak, pop_size=100, mutation_prob=.1,
                                                                          max_attempts=15,
                                                                          max_iters=1000, curve=True, random_state=723)
    toc = timeit.default_timer()
    tot_ga += toc - tic

t_ga = tot_ga/5

tot_mm = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_mm, best_fitness_mm, fitness_curve_mm = mlrose.mimic(problemSak, pop_size=900, keep_pct=.05,
                                                                    max_attempts=5,
                                                                    max_iters=1000, curve=True, random_state=781)
    toc = timeit.default_timer()
    tot_mm += toc - tic

t_mm = tot_mm/5

x = 'Sim Annealing (' + str(int(t_sa * 1000)) + ' ms)'
y = 'Random Hill (' + str(int(t_rh * 1000)) + ' ms)'
w = 'Genetic (' + str(int(t_ga * 1000)) + ' ms)'
z = 'Mimic (' + str(int(t_mm * 1000)) + ' ms)'
fig = pyplot.figure(figsize=(12,9))
pyplot.plot(range(len(fitness_curve_rh)), fitness_curve_rh,'b-o', label=y)
pyplot.plot(range(len(fitness_curve_sa)), fitness_curve_sa,'y-3', label=x)
pyplot.plot(range(len(fitness_curve_ga)), fitness_curve_ga,'r-+', label=w)
pyplot.plot(range(len(fitness_curve_mm)), fitness_curve_mm,'g-x', label=z)
pyplot.legend()
pyplot.xlim(0,550)
pyplot.ylim(0,190)
pyplot.xlabel('Iterations')
pyplot.title('napsack problem')
pyplot.ylabel('Fitness (Max 187)')
fig.savefig("napsackit.png")

# make iterations to maximize fitness vs problem size graph

iters_rh = []
iters_sa = []
iters_ga = []
iters_mm = []
rang = [5,7,9,11,13,15]
best_fitts = []
z = 0
for i in rang:

    fitness = mlrose.Knapsack(weights[:i], values[:i])
    problemSak = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True, max_val=2)
    #get max first with genetic Algorithms

    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problemSak, pop_size=100, mutation_prob=.1,
                                                                          max_attempts=15,
                                                                          max_iters=1000, curve=True, random_state=723)

    iters_ga.append(len(fitness_curve_ga))
    best_fitts.append(best_fitness_ga)

    #rh
    best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problemSak, max_iters=1000,
                                                                                restarts=1000,
                                                                                max_attempts=20,
                                                                                init_state=np.array([0] * i),
                                                                                curve=True,
                                                                                random_state=239)
    iters_rh.append(len(fitness_curve_rh))
    if best_fitness_rh < best_fitts[z]:
        print(i)
    


    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problemSak, schedule=mlrose.GeomDecay(
                                                                                        init_temp=10, decay=.998),
                                                                                  max_attempts=175, max_iters=10000,
                                                                                  init_state=np.array([0] * i),
                                                                                  curve=True, random_state=439)

    if best_fitness_sa < best_fitts[z]:
        print(i)
    else:
        iters_sa.append(len(fitness_curve_sa)-150)
    


    best_state_mm, best_fitness_mm, fitness_curve_mm = mlrose.mimic(problemSak, pop_size=900, keep_pct=.05,
                                                                    max_attempts=5,
                                                                    max_iters=1000, curve=True, random_state=781)


    if best_fitness_mm < best_fitts[z]:
        print(i)
    else:
        iters_mm.append(len(fitness_curve_mm))
    z += 1

x = 'Sim Annealing'
y = 'Random Hill'
w = 'Genetic'
z = 'Mimic'
fig = pyplot.figure(figsize=(12, 9))
pyplot.plot(rang, iters_rh, 'b-o', label=y)
pyplot.plot([5,11,13], iters_sa, 'y-3', label=x)
pyplot.plot(rang, iters_ga, 'r-+', label=w)
pyplot.plot(rang, iters_mm, 'g-x', label=z)
pyplot.legend()
pyplot.xlim(5, 15)
pyplot.ylim(0, 1000)
pyplot.xlabel('Input size')
pyplot.title('Knapsack')
pyplot.ylabel('Iterations to maximize')
fig.savefig("napsacksize.png")
