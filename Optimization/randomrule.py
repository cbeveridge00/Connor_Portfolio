
import mlrose_hiive as mlrose
import numpy as np
from matplotlib import pyplot
import timeit

#Long rule problem
def randomRule_max(state):
    rule = '0110000110010000111101001010000100101110101111000010100000100000110000100011100010010101100011110101101' \
           '0001110011101010110010110011000111001111101100111011011101000010111010110001001001010101110101110000' \
           '1111110001010011001111110001011110001011100001100000011010110110000000101101111001000011001100001111' \
           '1111110101000110000010000011101111001111010000000000100011110110011001101000110110100100011110010'

    rule = list(rule)
    fitcount = 0
    for i in range(len(state)):

        if int(state[i]) == int(rule[i]):
            fitcount += 1


    return fitcount


fitness = mlrose.CustomFitness(randomRule_max)
problem = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val = 2)

'''
best_statef = None
fitness_curve_rh = None
best_restart = 0
best_fitness_f = 0

#do randomhill gridsearch
for j in [1, 3, 5,25,60,100,150]:


    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_iters=1000,
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
for i in [.99,.98,.97,.96,.94,.93,.92,.91]:
    for j in [.1,.25,.5,1,2,3,5]:

        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
                                                                             schedule = mlrose.GeomDecay(init_temp=j, decay=i),
                                                                             max_iters=1000,
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
pop = [20,30,40,50,75]
mut = [.4,.5,.6,.7,.8]
for j in pop:
    for k in mut:

        tic = timeit.default_timer()
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=j, mutation_prob=k,
                                                                     max_iters=1000, curve=True,
                                                                     random_state=723)
        toc = timeit.default_timer()
        tot = toc - tic
        print('.')
        if best_fitness == 50 and tot < best_t:
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
for j in [50,100,140,200]:
    for k in [.1, .2, .35]:
        tic = timeit.default_timer()
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=j, keep_pct=k,
                                                                max_iters=1000, curve=True, random_state=781)

        toc = timeit.default_timer()
        tot = toc - tic
        print('.')
        if best_fitness==50 and tot < best_t:
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

#figure clock wall times
tot_rh = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                                restarts=100,
                                                                                init_state=None, curve=True,
                                                                                max_attempts=15,
                                                                                random_state=239)
    toc = timeit.default_timer()
    tot_rh += toc - tic

t_rh = tot_rh/5

tot_sa = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem,
                                                                              schedule = mlrose.GeomDecay(init_temp=2, decay=.98),
                                                                              max_iters=1000,
                                                                              max_attempts=40,
                                                                              curve=True, random_state=439)
    toc = timeit.default_timer()
    tot_sa += toc - tic

t_sa = tot_sa/5

tot_ga = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=30, mutation_prob=.5,
                                                                          max_iters=1000, curve=True,
                                                                          random_state=723)
    toc = timeit.default_timer()
    tot_ga += toc - tic

t_ga = tot_ga/5

tot_mm = 0
for i in range(5):
    tic = timeit.default_timer()
    best_state, best_fitness, fitness_curve_mm = mlrose.mimic(problem, pop_size=200, keep_pct=.1, max_attempts=5,
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
pyplot.xlim(0,250)
pyplot.ylim(25,51)
pyplot.xlabel('Iterations')
pyplot.title('Random Rule')
pyplot.ylabel('Fitness (max 50)')
fig.savefig("randomrulecomp.png")


#make iterations to maximize fitness vs problem size graph

iters_rh = []
iters_sa = []
iters_ga = []
iters_mm = []
rang = [10,20,30,40,50,60,70,80,90]
for i in rang:


    problem = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True, max_val=2)

    if i < 80:
        ma = 15
    else:
        ma = 18
    best_state_rh, best_fitness_rh, fitness_curve_rh = mlrose.random_hill_climb(problem, max_iters=1000,
                                                                                restarts=100,
                                                                                init_state=None, curve=True,
                                                                                max_attempts=ma,
                                                                                random_state=239)
    iters_rh.append(len(fitness_curve_rh))
    

    if i < 60:
        ma = 70
        tract = 60
    elif i < 40:
        ma = 10
        tract = 0
    else:
        ma = 125
        tract = 115
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem,
                                                                                  schedule=mlrose.GeomDecay(init_temp=2,
                                                                                                            decay=.98),
                                                                                  max_iters=1000,
                                                                                  max_attempts=ma,
                                                                                  curve=True, random_state=439)
    iters_sa.append(len(fitness_curve_sa) - tract)
    if best_fitness_sa != i:
        print(i)



    if i < 70:
        ma = 10
        tract = 0

    else:
        ma = 55
        tract = 45
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=30, mutation_prob=.5,
                                                                          max_iters=1000, curve=True,
                                                                          max_attempts=ma,
                                                                          random_state=723)

    iters_ga.append(len(fitness_curve_ga) - tract)
    if best_fitness_ga != i:
        print(i)


    ma = 5
    tract = 0
    kp = .1
    if i == 40:
        ma = 4
    elif i < 40:
        ma=3

    if i > 69:
        ma = 10
        tract = 5
        kp = .15

    best_state, best_fitness, fitness_curve_mm = mlrose.mimic(problem, pop_size=200, keep_pct=kp, max_attempts=ma,
                                                              max_iters=1000, curve=True, random_state=781)


    iters_mm.append(len(fitness_curve_mm)-tract)
    if best_fitness != i:
        print(i)

x = 'Sim Annealing'
y = 'Random Hill'
w = 'Genetic'
z = 'Mimic'
fig = pyplot.figure(figsize=(12,9))
pyplot.plot(rang, iters_rh,'b-o', label=y)
pyplot.plot(rang, iters_sa,'y-3', label=x)
pyplot.plot(rang, iters_ga,'r-+', label=w)
pyplot.plot(rang, iters_mm,'g-x', label=z)
pyplot.legend()
pyplot.xlim(0,100)
pyplot.ylim(0,500)
pyplot.xlabel('Input size')
pyplot.title('Random Rule')
pyplot.ylabel('Iterations to maximize')
fig.savefig("randomrulesize.png")
