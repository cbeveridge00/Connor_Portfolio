import mlrose_hiive as mlrose
import numpy as np
from matplotlib import pyplot
import timeit
import time

# Long rule problem
def love7Rule_max(state):

    fitcount = 0
    #time.sleep(60)
    for i in range(len(state)):

        if int(state[i]) == 7:
            fitcount += 1

    return fitcount


fitness = mlrose.CustomFitness(love7Rule_max)
problem = mlrose.DiscreteOpt(length=10, fitness_fn=fitness, maximize=True, max_val=10)

'''
best_statef = None
fitness_curve_rh = None
best_restart = 0
best_fitness_f = 0

#do randomhill gridsearch
for j in [5,10,15,20,25,35]:


    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                       max_attempts=45,
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

best_fitness_sa = 0
best_state_sa = None
fitness_curve_sa = None
best_decay_sa = 0
best_temp = 0

#do simulated annealing gridsearch
for i in [.995,.99,.98,.97,.96,.94]:
    for j in [.5,1,2,3,5,10]:

        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
                                                                             schedule = mlrose.GeomDecay(init_temp=j, decay=i),
                                                                             max_iters=1000,
                                                                             max_attempts=10,
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
pop = [5,10,20,30,40,50]
mut = [.4,.5,.6,.7,.8]
for j in pop:
    for k in mut:

        tic = timeit.default_timer()
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=j, mutation_prob=k,
                                                                     max_iters=1000, curve=True,
                                                                     random_state=723)
        toc = timeit.default_timer()
        tot = toc - tic

        if best_fitness == 10 and tot < best_t:
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





#Do mimic gridsearch
best_fitness_mm = 0
best_state_mm = None
fitness_curve_mm = None
best_pop = 0
best_pct = 0
tot = 0
best_t = 100000
for j in [400,500,600,700,800]:
    for k in [.025,.05,.1, .2]:
        tic = timeit.default_timer()
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=j, keep_pct=k,
                                                                max_iters=1000, curve=True, random_state=781)

        toc = timeit.default_timer()
        tot = toc - tic
        print('.')
        if best_fitness==10 and tot < best_t:
            best_fitness_mm = best_fitness
            best_state_mm = best_state
            fitness_curve_mm = fitness_curve
            best_pop = j
            best_pct = k
            best_t = tot

print('Mimic Best:')
print(best_fitness_mm)
print(best_state_mm)
print(best_pop)
print(best_pct)

'''

# figure clock wall times

best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                            restarts=100,
                                                                            init_state=None, curve=True,
                                                                            max_attempts=45,
                                                                            random_state=239)

best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem,
                                                                              schedule=mlrose.GeomDecay(init_temp=5,
                                                                                                        decay=.99),
                                                                              max_iters=1000,
                                                                              curve=True, random_state=439)

best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=10, mutation_prob=.6,
                                                                      max_iters=1000, curve=True,
                                                                      random_state=723)

best_state, best_fitness, fitness_curve_mm = mlrose.mimic(problem, pop_size=600, keep_pct=.025, max_attempts=2,
                                                              max_iters=1000, curve=True, random_state=781)


x = 'Sim Annealing (' + str(len(fitness_curve_rh)) + ' mins)'
y = 'Random Hill (' + str(len(fitness_curve_sa)) + ' mins)'
w = 'Genetic (' + str(len(fitness_curve_ga)) + ' mins)'
z = 'Mimic (' + str(len(fitness_curve_mm)) + ' mins)'
fig = pyplot.figure(figsize=(12, 9))
pyplot.plot(range(len(fitness_curve_rh)), fitness_curve_rh, 'b-o', label=y)
pyplot.plot(range(len(fitness_curve_sa)), fitness_curve_sa, 'y-3', label=x)
pyplot.plot(range(len(fitness_curve_ga)), fitness_curve_ga, 'r-+', label=w)
pyplot.plot(range(len(fitness_curve_mm)), fitness_curve_mm, 'g-x', label=z)
pyplot.legend()
pyplot.xlim(-15, 350)
pyplot.ylim(0, 10)
pyplot.xlabel('Iterations')
pyplot.title('Love 7s')
pyplot.ylabel('Fitness (max 50)')
fig.savefig("love7comp.png")


# make iterations to maximize fitness vs problem size graph

iters_rh = []
iters_sa = []
iters_ga = []
iters_mm = []
rang = [4,6,8,10,12,14,16]
for i in rang:

    problem = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True, max_val=10)



    best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                                restarts=100,
                                                                                init_state=None, curve=True,
                                                                                max_attempts=45,
                                                                                random_state=239)
    iters_rh.append(len(fitness_curve_rh))
    if best_fitness_rh != i:
        print(i)
        
    
    if i < 8:
        ma = 10
        tract = 0
    elif i > 10:
        ma = 35
        tract = 25
    else:
        ma = 15
        tract = 5
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem,
                                                                              schedule=mlrose.GeomDecay(init_temp=5,
                                                                                                        decay=.99),
                                                                              max_iters=1000,
                                                                                max_attempts=ma,
                                                                              curve=True, random_state=439)
    iters_sa.append(len(fitness_curve_sa) -tract)
    if best_fitness_sa != i:
        print(i)


    ma = 10
    tract = 0
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=10, mutation_prob=.6,
                                                                      max_iters=1000, curve=True,
                                                                      random_state=723)

    iters_ga.append(len(fitness_curve_ga) - tract)
    if best_fitness_ga != i:
        print(i)

    best_state, best_fitness, fitness_curve_mm = mlrose.mimic(problem, pop_size=600, keep_pct=.075,
                                                              max_iters=1000, curve=True, random_state=781)

    iters_mm.append(len(fitness_curve_mm) )
    if best_fitness != i:
        print(i)
        print(best_state, best_fitness)

x = 'Sim Annealing'
y = 'Random Hill'
w = 'Genetic'
z = 'Mimic'
fig = pyplot.figure(figsize=(12, 9))
pyplot.plot(rang, iters_rh,'b-o', label=y)
pyplot.plot(rang, iters_sa,'y-3', label=x)
pyplot.plot(rang, iters_ga,'r-+', label=w)
pyplot.plot(rang, iters_mm, 'g-x', label=z)
pyplot.legend()
pyplot.xlim(4,16)
pyplot.ylim(0, 800)
pyplot.xlabel('Input size')
pyplot.title('Love 7s')
pyplot.ylabel('Iterations to maximize')
fig.savefig("love7size.png")


