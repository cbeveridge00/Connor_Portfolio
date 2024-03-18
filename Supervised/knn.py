from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas
from sklearn.preprocessing import LabelEncoder
import timeit
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot
from curve import plot_learning_curve

#use pandas to get data in a nice format
game_data = pandas.read_csv("input/tic-tac-toe.csv")
heart_data = pandas.read_csv("input/heart.csv")

#Prepare data and split into testing and training sets
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
              'weights': ('uniform', 'distance'),
              'p':[1,2],
             }


KNN = KNeighborsClassifier()
gsearch1 = GridSearchCV(KNN, param_grid = parameters, cv = 15, scoring='accuracy')
gsearch1.fit(X_train,y_train)

KNN2 = KNeighborsClassifier()
gsearch2 = GridSearchCV(KNN2, param_grid = parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart,y_train_heart)

DT_best_crite = gsearch1.best_params_['weights']
print("For a TTT Decision Tree model, the optimal weighting is is "+str(DT_best_crite))
DT_best_split = gsearch1.best_params_['p']
print(", the optimal p is "+str(DT_best_split))

DT_best_crite = gsearch2.best_params_['weights']
print("For a TTT Decision Tree model, the optimal weighting is is "+str(DT_best_crite))
DT_best_split = gsearch2.best_params_['p']
print(", the optimal p is "+str(DT_best_split))

model = gsearch1.best_estimator_
score = model.score(X_test, y_test)
print('Initial OptimalModel Score (TTT):')
print(score)

model2 = gsearch2.best_estimator_
score = model2.score(X_test_heart, y_test_heart)
print('Initial Optimal Model Score (HD):')
print(score)

fig4, axes = pyplot.subplots()
fig5, axes2 = pyplot.subplots()

title = "Learning Curve (Tic-Tac-Toe)"
title2 = "Learning Curve (Heart Disease)"
# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=168)

cv2 = ShuffleSplit(n_splits=15, test_size=0.2, random_state=789)

estimator = KNeighborsClassifier(weights='uniform', p=1)

estimator2 = KNeighborsClassifier(weights='distance', p=2)

plot_learning_curve(estimator, title, X_train, y_train, axes, ylim=(0.55, 1.01),
                    cv=cv)

plot_learning_curve(estimator2, title2, X_train_heart, y_train_heart, axes2, ylim=(0.55, 1.01),
                    cv=cv2)

fig4.savefig("fig7a.png")
fig5.savefig("fig7b.png")


#Model Complexity Anlysis
parameters1 = {
              'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              'weights': ['uniform'],
              'p':[1],
             }

parameters2 = {
              'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              'weights': ['distance'],
              'p':[2],
             }
gsearch1 = GridSearchCV(KNN, parameters1, scoring='accuracy')
gsearch1.fit(X_train, y_train)

gsearch2 = GridSearchCV(KNN2, parameters2, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

DT_best_layers = gsearch1.best_params_['n_neighbors']
print("For a TTT KNN model, the optimal number of k is "+str(DT_best_layers))

DT_best_layers = gsearch2.best_params_['n_neighbors']
print("For a heart KNN model, the optimal number of k is "+str(DT_best_layers))

DT_df = pandas.DataFrame(gsearch1.cv_results_)
fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_n_neighbors'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(0,20)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('k')
pyplot.ylabel('Mean Accuracy 15-fold CV')
pyplot.title("Tic-Tac-Toe Endgame")
fig2.savefig("fig8a.png")

DT_df = pandas.DataFrame(gsearch2.cv_results_)
fig3 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_n_neighbors'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(0,20)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('k')
pyplot.ylabel('Mean Accuracy 15-fold CV')
pyplot.title("Heart Disease")
fig3.savefig("fig8b.png")

#optimal learner

fig6, axes3 = pyplot.subplots()

title = "Learning Curve optimal (Tic-Tac-Toe)"
# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=168)

estimator = KNeighborsClassifier(n_neighbors=12, weights='uniform', p=1)

plot_learning_curve(estimator, title, X_train, y_train, axes3, ylim=(0.55, 1.01),
                    cv=cv)

fig6.savefig("fig9b.png")

parametersf = {
              'n_neighbors': [12],
              'weights': ['uniform'],
              'p':[1],
             }

KNN = KNeighborsClassifier()
gsearchff = GridSearchCV(KNN, param_grid = parametersf, cv = 15, scoring='accuracy')
gsearchff.fit(X_train,y_train)
model = gsearchff.best_estimator_
score = model.score(X_test, y_test)
print(score)

#Time optimized learner
def optumTTT():
    a = KNeighborsClassifier(n_neighbors=12, weights='uniform', p=1)
    a.fit(X_train, y_train)
    b = a.score(X_test, y_test)
    return b


def optumHD():
    a = KNeighborsClassifier(n_neighbors=10, weights='distance', p=2)
    a.fit(X_train_heart, y_train_heart)
    b = a.score(X_test_heart, y_test_heart)
    return b

print('Final Testing Accuracies:')
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