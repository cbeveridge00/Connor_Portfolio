import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlrose_hiive as mlrose
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot
import timeit
game_data = pandas.read_csv("input/tic-tac-toe.csv")

#labelencode the values with onehot encoding




x = game_data.iloc[:, 0:9]
y = game_data.iloc[:, 9]

encoder = LabelEncoder()
y = encoder.fit_transform(y)
x = pandas.get_dummies(x)

X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=200, train_size=758, random_state=212)

X_val = X_train_full[600:758]
X_train = X_train_full[:600]

y_val = y_train_full[600:758]
y_train = y_train_full[:600]

#run neural network for gradient descent to replicate assignment 1 results
nn_gd = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 450, \
                                 bias = True, is_classifier = True, learning_rate = 0.0025, \
                                 clip_max = 1, curve=True, \
                                 random_state = 852)

nn_gd.fit(X_train, y_train)

#train accuracy
y_train_pred = nn_gd.predict(X_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print(y_train_accuracy)
#print(nn_gd.fitness_curve[1])

#test accuracy
y_test_pred = nn_gd.predict(X_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)
'''
# Initialize neural network object rh and fit object and do gridsearch
best_acc = 0
best_curve = []
best_lr = 0
for i in [2,5,10]:
    nn_rh = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 3000, \
                                     bias = True, is_classifier = True, restarts=i, \
                                     early_stopping = True, clip_max = 1, curve=True, max_attempts = 100, \
                                     random_state = 852)


    nn_rh.fit(X_train, y_train)


    #test accuracy
    y_test_pred = nn_rh.predict(X_val)

    y_test_accuracy = accuracy_score(y_val, y_test_pred)
    if y_test_accuracy > best_acc:
        best_acc = y_test_accuracy
        best_curve = nn_rh.fitness_curve
        best_lr = i

print(best_acc)
print(best_lr)


# Initialize neural network sa object and fit object and do gridsearch for sa
best_acc = 0
best_curve = []
for i in [.5,1,2,5]:
    for j in [.995,.99,.95]:
        nn_sa = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 10000, \
                                 bias = True, is_classifier = True, \
                                 early_stopping = True, clip_max = 1, curve=True, max_attempts = 100, \
                                 schedule=mlrose.GeomDecay(init_temp=i, decay=j), random_state = 852)

        nn_sa.fit(X_train, y_train)


        #test accuracy
        y_test_pred = nn_sa.predict(X_val)

        y_test_accuracy = accuracy_score(y_val, y_test_pred)

        if y_test_accuracy > best_acc:
            print(y_test_accuracy)
            best_acc = y_test_accuracy
            best_curve=nn_sa.fitness_curve
            print(i,j)



# Initialize neural network sa object and fit object and do gridsearch for genetic alg
best_acc = 0
best_curve_ga = []
for i in [100,150,200]:
    for j in [.1,.2,.3]:
        nn_ga = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, \
                                 early_stopping = True, clip_max = 1, curve=True, max_attempts = 40, \
                                 mutation_prob=j, pop_size=i, random_state = 852)

        nn_ga.fit(X_train, y_train)


        #test accuracy
        y_test_pred = nn_ga.predict(X_val)

        y_test_accuracy = accuracy_score(y_val, y_test_pred)

        if y_test_accuracy > best_acc:
            print(y_test_accuracy)
            best_acc = y_test_accuracy
            best_curve_ga =nn_ga.fitness_curve
            print(i,j)
            


'''

#find best times to converge and final accuracies and graph
#rh
tic = timeit.default_timer()
nn_rh = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 12000, \
                                     bias = True, is_classifier = True, \
                                     early_stopping = True, clip_max = 1, curve=True, max_attempts = 100, \
                                     restarts = 10, random_state = 852)


nn_rh.fit(X_train_full, y_train_full)
#test accuracy
y_test_pred = nn_rh.predict(X_test)

print(accuracy_score(y_test, y_test_pred))
toc = timeit.default_timer()
tot_rh = toc - tic

#sa
tic = timeit.default_timer()
nn_sa = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 15000, \
                                 bias = True, is_classifier = True, \
                                 early_stopping = True, clip_max = 1, curve=True, max_attempts = 100, \
                                 schedule=mlrose.GeomDecay(init_temp=.5), random_state = 852)

nn_sa.fit(X_train_full, y_train_full)
#test accuracy
y_test_pred = nn_sa.predict(X_test)

print(accuracy_score(y_test, y_test_pred))
toc = timeit.default_timer()
tot_sa = toc - tic

#genetic
tic = timeit.default_timer()
nn_ga = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 2000, \
                                 bias = True, is_classifier = True, \
                                 early_stopping = True, clip_max = 1, curve=True, max_attempts = 250, \
                                 mutation_prob=.1, pop_size=100, random_state = 852)

nn_ga.fit(X_train_full, y_train_full)
#test accuracy
y_test_pred = nn_ga.predict(X_test)

print(accuracy_score(y_test, y_test_pred))
toc = timeit.default_timer()
tot_ga = toc - tic

x = 'Sim Annealing (' + str(int(tot_sa)) + ' ms)'
y = 'Random Hill(' + str(int(tot_rh)) + ' ms)'
w = 'Genetic(' + str(int(tot_ga)) + ' ms)'
fig = pyplot.figure(figsize=(12, 9))
pyplot.plot(range(len(nn_rh.fitness_curve)), nn_rh.fitness_curve, 'b-o', label=y)
pyplot.plot(range(len(nn_sa.fitness_curve)), nn_sa.fitness_curve, 'g-x', label=x)
pyplot.plot(range(len(nn_ga.fitness_curve)), nn_ga.fitness_curve, 'r-+', label=w)
pyplot.legend()
pyplot.xlim(0, 15000)
pyplot.ylim(0, 1.25)
pyplot.xlabel('Iterations')
pyplot.title('Tic-Tac-Toe Endgame ANN')
pyplot.ylabel('Loss')
fig.savefig("neuralfit.png")




#get accuracies vs iterations
best_acc = 0.0
acc_sa = []
best_t = 0
its = [1000,2000,5000,10000,20000,30000,40000]
for i in its:
    tic = timeit.default_timer()
    nn_sa = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                     algorithm = 'simulated_annealing', max_iters = i, \
                                     bias = True, is_classifier = True, \
                                     early_stopping = True, clip_max = 1, curve=True, max_attempts = 100, \
                                     schedule=mlrose.GeomDecay(init_temp=.5), random_state = 852)

    nn_sa.fit(X_train_full, y_train_full)
    #test accuracy
    y_test_pred = nn_sa.predict(X_test)
    score = accuracy_score(y_test, y_test_pred)
    acc_sa.append(score)
    toc = timeit.default_timer()
    tot_sa = toc - tic
    print(score)
    if score > best_acc:
        best_t = tot_sa
        best_acc = score

print(best_t)
x = 'Sim Annealing (' + str(int(best_t)) + ' s) at peak)'

best_acc = 0.0
acc_rh = []
best_t = 0
its = [1000,2000,5000,10000,20000,30000,40000]
for i in its:
    tic = timeit.default_timer()
    nn_rh = mlrose.NeuralNetwork(hidden_nodes=[8], activation='relu', \
                                 algorithm='random_hill_climb', max_iters=i, \
                                 bias=True, is_classifier=True, \
                                 early_stopping=True, clip_max=1, curve=True, max_attempts=100, \
                                 restarts=10, random_state=852)

    nn_rh.fit(X_train_full, y_train_full)
    #test accuracy
    y_test_pred = nn_rh.predict(X_test)
    score = accuracy_score(y_test, y_test_pred)
    acc_rh.append(score)
    toc = timeit.default_timer()
    tot_rh = toc - tic
    print(score)
    if score > best_acc:
        best_t = tot_rh
        best_acc = score

print(best_t)
y = 'Random Hill(' + str(int(best_t)) + ' s at peak)'
#w = 'Genetic(' + str(int(tot_ga)) + ' s)'
fig = pyplot.figure(figsize=(12, 9))
pyplot.plot(its, acc_sa, 'g-o', label=x)
pyplot.plot(its, acc_rh, 'b-x', label=y)
#pyplot.plot(range(len(nn_ga.fitness_curve)), nn_ga.fitness_curve, 'r-+', label=w)
pyplot.legend()
pyplot.xlim(0, 40000)
pyplot.ylim(0, 1)
pyplot.xlabel('Iterations')
pyplot.title('Tic-Tac-Toe Endgame ANN')
pyplot.ylabel('Accuracy')
fig.savefig("neuralfitacc.png")


#find best accuracy vs maximum iterations - ga

best_acc = 0.0
acc_ga = []
best_t = 0
its = [200,400,750,1500,2500]
for i in its:
    tic = timeit.default_timer()
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=[8], activation='relu', \
                                 algorithm='genetic_alg', max_iters=i, \
                                 bias=True, is_classifier=True, \
                                 early_stopping=True, clip_max=1, curve=True, max_attempts=500, \
                                 mutation_prob=.1, pop_size=100, random_state=852)

    nn_ga.fit(X_train_full, y_train_full)
    #test accuracy
    y_test_pred = nn_ga.predict(X_test)
    score = accuracy_score(y_test, y_test_pred)
    acc_ga.append(score)
    toc = timeit.default_timer()
    tot_ga = toc - tic
    print(score)
    if score > best_acc:
        best_t = tot_ga

print(best_t)
w = 'Genetic(' + str(int(best_t)) + ' s at peak)'

#find best accuracy vs maximum iterations - gd
best_acc = 0.0
acc_gd = []
best_t = 0
its2 = [50,100,200,400,750,1500,2400]
for i in its2:
    tic = timeit.default_timer()
    nn_gd = mlrose.NeuralNetwork(hidden_nodes = [8], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = i, \
                                 bias = True, is_classifier = True, learning_rate = 0.0025, \
                                 clip_max = 1, curve=True, \
                                 random_state = 852)

    nn_gd.fit(X_train_full, y_train_full)
    #test accuracy
    y_test_pred = nn_gd.predict(X_test)
    score = accuracy_score(y_test, y_test_pred)
    acc_gd.append(score)
    toc = timeit.default_timer()
    tot_gd = toc - tic
    print(score)
    if score > best_acc:
        best_t = tot_gd

print(best_t)
d = 'Gradient Descent(' + str(int(best_t)) + ' s at peak)'



fig = pyplot.figure(figsize=(12, 9))
pyplot.plot(its2, acc_gd, 'k-x', label=d)
pyplot.plot(its, acc_ga, 'r-+', label=w)
pyplot.legend()
pyplot.xlim(0, 2500)
pyplot.ylim(0, 1)
pyplot.xlabel('Max Iterations')
pyplot.title('Tic-Tac-Toe Endgame ANN')
pyplot.ylabel('Accuracy')
fig.savefig("neuralfitacc2.png")
