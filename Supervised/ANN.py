import pandas
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
import timeit
from sklearn.metrics import classification_report


#use pandas to get data in a nice format
game_data = pandas.read_csv("input/tic-tac-toe.csv")
heart_data = pandas.read_csv("input/heart.csv")


#One feature is dropped which only has 1 option so is useless
#game_data.drop('veil-type', axis=1, inplace=True)
#game_data.drop('stalk-root', axis=1, inplace=True)

#labelencode the values with onehot encoding




x = game_data.iloc[:, 0:9]
y = game_data.iloc[:, 9]

heart_x = heart_data.iloc[:, 0:13]
heart_y = heart_data.iloc[:, 13]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

one_hot = pandas.get_dummies(heart_x['cp'])
heart_x = heart_x.drop('cp',axis = 1)
heart_x = heart_x.join(one_hot)

x = pandas.get_dummies(x)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=200, train_size=758, random_state=212)
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(heart_x, heart_y, test_size=200,
                                                                            train_size=758, random_state=354)


#Inital gridsearch - tictactoe
parameters = {

              'learning_rate_init': [.0005, .0008, .001, .0015, .002, .0025],
              'momentum': [0, .25, .5, .75, 1]
             }

mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(3), max_iter=750, random_state=852)
mlp2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(3), max_iter=750, random_state=123)

gsearch = GridSearchCV(mlp, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

DT_best_crite = gsearch.best_params_['learning_rate_init']
print("For a TTT ANN model, the optimal rate is "+str(DT_best_crite))
DT_best_crite = gsearch.best_params_['momentum']
print("For a TTT ANN model, the optimal momentum is "+str(DT_best_crite))
model = gsearch.best_estimator_
score = model.score(X_test, y_test)
print(score)


DT_best_crite = gsearch2.best_params_['learning_rate_init']
print("For a heart ANN model, the optimal rate is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['momentum']
print("For a TTT ANN model, the optimal momentum is "+str(DT_best_crite))
model2 = gsearch2.best_estimator_
score2 = model2.score(X_test_heart, y_test_heart)
print(score2)



#model complexity analysis
parameters = {
              'hidden_layer_sizes': [(1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13)],
             }

mlp = MLPClassifier(solver='sgd', momentum=1, learning_rate_init=.0025, max_iter=750, random_state=852)
mlp2 = MLPClassifier(solver='sgd', momentum=0.75, learning_rate_init=.0008, max_iter=750, random_state=123)

gsearch = GridSearchCV(mlp, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_
score = model.score(X_test, y_test)
print('optimal model score(TTT):')
print(score)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

model = gsearch2.best_estimator_
score = model.score(X_test_heart, y_test_heart)
print('optimal model score(HD):')
print(score)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a TTT ANN model, the optimal units are "+str(DT_best_crite))


DT_best_crite = gsearch2.best_params_['hidden_layer_sizes']
print("For a heart ANN model, the optimal units are "+str(DT_best_crite))

DT_df = pandas.DataFrame(gsearch.cv_results_)

fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_hidden_layer_sizes'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(1,13)
pyplot.ylim(0.45,1.0)
pyplot.xlabel('Hidden Layer Untis (1-layer)')
pyplot.title('Tic-Tac-Toe Endgame')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig2.savefig("fig3a.png")

DT_df = pandas.DataFrame(gsearch2.cv_results_)

fig3 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_hidden_layer_sizes'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(1,13)
pyplot.ylim(0.45,1.0)
pyplot.xlabel('Hidden Layer Untis (1-layer)')
pyplot.title('Heart Disease')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig3.savefig("fig3b.png")


#Do iterative Learning Curve

parameters = {
                'max_iter': [15, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800]
             }

mlp = MLPClassifier(hidden_layer_sizes=(11), solver='sgd', momentum=1, learning_rate_init=.0025, random_state=852)
mlp2 = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, learning_rate_init=.0008, random_state=123)

gsearch = GridSearchCV(mlp, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_
score = model.score(X_test, y_test)
print(score)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

model = gsearch2.best_estimator_
score = model.score(X_test_heart, y_test_heart)
print(score)

DT_best_crite = gsearch.best_params_['max_iter']
print("For a TTT ANN model, the optimal max_iter are "+str(DT_best_crite))


DT_best_crite = gsearch2.best_params_['max_iter']
print("For a heart ANN model, the optimal units are "+str(DT_best_crite))

DT_df = pandas.DataFrame(gsearch.cv_results_)

fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_max_iter'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(15,800)
pyplot.ylim(0.45,1.0)
pyplot.xlabel('Iterations')
pyplot.title('Tic-Tac-Toe Endgame')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig2.savefig("fig4a.png")

DT_df = pandas.DataFrame(gsearch2.cv_results_)

fig3 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_max_iter'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(15,800)
pyplot.ylim(0.45,1.0)
pyplot.xlabel('Iterations')
pyplot.title('Heart-Disease')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig3.savefig("fig4b.png")


#Time optimized learner
def optumTTT():
    a = MLPClassifier(hidden_layer_sizes=(11), solver='sgd', momentum=1, max_iter=250, learning_rate_init=.0025,
                      random_state=852)
    a.fit(X_train, y_train)
    b = a.score(X_test, y_test)
    return b


def optumHD():
    a = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=250, learning_rate_init=.0008,
                      random_state=123)
    a.fit(X_train_heart, y_train_heart)
    b = a.score(X_test_heart, y_test_heart)
    return b

print('Accuracies:')
print(optumTTT())
print(optumHD())

print('times (ms):')
tot = 0
for _ in range(10):
    tic=timeit.default_timer()
    optumTTT()
    toc=timeit.default_timer()
    tot += toc-tic


print((tot/10.0)*1000)

tot = 0
for _ in range(10):
    tic=timeit.default_timer()
    optumHD()
    toc=timeit.default_timer()
    tot += toc-tic

print((tot/10.0)*1000)