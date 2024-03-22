from sklearn.random_projection import SparseRandomProjection
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
import timeit
from matplotlib import pyplot

#use pandas to get data in a nice format
game_data = pandas.read_csv("input/tic-tac-toe.csv")
heart_data = pandas.read_csv("input/heart.csv")

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

heart_x = StandardScaler().fit_transform(heart_x)
x = pandas.get_dummies(x)
x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=200, train_size=758, random_state=212)
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(heart_x, heart_y, test_size=200,
                                                                            train_size=758, random_state=354)

#run all DRs on the heart dataset

#pca
pca = PCA(random_state=10)
principalComponentsfit = pca.fit(X_train_heart)
principalComponents = np.array(principalComponentsfit.transform(X_train_heart))
principalComponents = principalComponents[:,:11]
principalComponentstest = np.array(principalComponentsfit.transform(X_test_heart))
principalComponentstest = principalComponentstest[:,:11]

#ICA
ica = FastICA(n_components=8, random_state=10, max_iter=3000)
ica_fit = ica.fit(X_train_heart)
ica_ = ica_fit.transform(X_train_heart)
ica_test = ica_fit.transform(X_test_heart)


#RP
rp = SparseRandomProjection(n_components=13, random_state=10)
rp_fit = rp.fit(X_train_heart)
rp_ = rp_fit.transform(X_train_heart)
rp_test = rp_fit.transform(X_test_heart)

#FA
fa = FeatureAgglomeration(n_clusters=3, linkage='average')
heart_red_fit = fa.fit(X_train_heart)
heart_red = heart_red_fit.transform(X_train_heart)
heart_red_test = heart_red_fit.transform(X_test_heart)

a = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=1000, learning_rate_init=.0008,
                      random_state=123)

a.fit(X_train_heart, y_train_heart)
print(a.score(X_test_heart, y_test_heart))

'''
#Do gridsearch for each DR for best parameters
parameters = {

              'learning_rate_init': [.0008, .001, .0015],
              'momentum': [0, .5, 1],
              'hidden_layer_sizes': [(3), (6), (9)]
             }

#PCA
mlp2 = MLPClassifier(solver='sgd', max_iter=750, random_state=123)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 10, scoring='accuracy')
gsearch2.fit(principalComponents, y_train_heart)

DT_best_crite = gsearch2.best_params_['learning_rate_init']
print("For a PCA heart ANN model, the optimal rate is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['momentum']
print("For a PCA heart ANN model, the optimal momentum is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['hidden_layer_sizes']
print("For a PCA heart ANN model, the optimal units is "+str(DT_best_crite))
model2 = gsearch2.best_estimator_
score2 = model2.score(principalComponentstest, y_test_heart)
print(score2)


#ICA
mlp2 = MLPClassifier(solver='sgd', max_iter=750, random_state=123)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 10, scoring='accuracy')
gsearch2.fit(ica_, y_train_heart)

DT_best_crite = gsearch2.best_params_['learning_rate_init']
print("For a ICA heart ANN model, the optimal rate is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['momentum']
print("For a ICA heart ANN model, the optimal momentum is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['hidden_layer_sizes']
print("For a ICA heart ANN model, the optimal units is "+str(DT_best_crite))
model2 = gsearch2.best_estimator_
score2 = model2.score(ica_test, y_test_heart)
print(score2)

#RP
mlp2 = MLPClassifier(solver='sgd', max_iter=750, random_state=123)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 10, scoring='accuracy')
gsearch2.fit(rp_, y_train_heart)

DT_best_crite = gsearch2.best_params_['learning_rate_init']
print("For a RP heart ANN model, the optimal rate is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['momentum']
print("For a Rp heart ANN model, the optimal momentum is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['hidden_layer_sizes']
print("For a RP heart ANN model, the optimal units is "+str(DT_best_crite))
model2 = gsearch2.best_estimator_
score2 = model2.score(rp_test, y_test_heart)
print(score2)

mlp2 = MLPClassifier(solver='sgd', max_iter=750, random_state=123)

gsearch2 = GridSearchCV(mlp2, parameters, cv = 10, scoring='accuracy')
gsearch2.fit(heart_red, y_train_heart)

DT_best_crite = gsearch2.best_params_['learning_rate_init']
print("For a PCA heart ANN model, the optimal rate is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['momentum']
print("For a PCA heart ANN model, the optimal momentum is "+str(DT_best_crite))
DT_best_crite = gsearch2.best_params_['hidden_layer_sizes']
print("For a PCA heart ANN model, the optimal units is "+str(DT_best_crite))
model2 = gsearch2.best_estimator_
score2 = model2.score(heart_red_test, y_test_heart)
print(score2)
'''

# do Iteraction with CV accuracy
#Do iterative Learning Curve
'''
parameters = {
                'max_iter': [25, 50, 75, 100, 200, 300, 400, 500, 600, 700]
             }

a = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=1000, learning_rate_init=.0008,
                      random_state=123)



mlpPCA= MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.5, learning_rate_init=.001, random_state=123)
mlpICA = MLPClassifier(hidden_layer_sizes=(3), solver='sgd', momentum=1, learning_rate_init=.0008, random_state=123)
mlpRP = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.5, learning_rate_init=.0015, random_state=123)
mlpFA = MLPClassifier(hidden_layer_sizes=(9), solver='sgd', momentum=0.5, learning_rate_init=.0008, random_state=123)

gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(X_train_heart, y_train_heart)
DT_Org = pandas.DataFrame(gsearch.cv_results_)

gsearchPCA = GridSearchCV(mlpPCA, parameters, cv = 10, scoring='accuracy')
gsearchPCA.fit(principalComponents, y_train_heart)
DT_PCA = pandas.DataFrame(gsearchPCA.cv_results_)

gsearchICA = GridSearchCV(mlpICA, parameters, cv = 10, scoring='accuracy')
gsearchICA.fit(ica_, y_train_heart)
DT_ICA = pandas.DataFrame(gsearchICA.cv_results_)

gsearchRP = GridSearchCV(mlpRP, parameters, cv = 10, scoring='accuracy')
gsearchRP.fit(rp_, y_train_heart)
DT_RP = pandas.DataFrame(gsearchRP.cv_results_)

gsearchFA = GridSearchCV(mlpFA, parameters, cv = 10, scoring='accuracy')
gsearchFA.fit(heart_red, y_train_heart)
DT_FA = pandas.DataFrame(gsearchFA.cv_results_)



fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_Org['param_max_iter'],DT_Org['mean_test_score'],'g-o', label='Original')
pyplot.plot(DT_PCA['param_max_iter'],DT_PCA['mean_test_score'],'b-x', label='PCA')
pyplot.plot(DT_ICA['param_max_iter'],DT_ICA['mean_test_score'],'r-+', label='ICA')
pyplot.plot(DT_RP['param_max_iter'],DT_RP['mean_test_score'],'c-*', label='RP')
pyplot.plot(DT_FA['param_max_iter'],DT_FA['mean_test_score'],'m-x', label='FA')
pyplot.xlim(20,710)
pyplot.ylim(0.5,.95)
pyplot.legend()
pyplot.xlabel('Iterations')
pyplot.title('Heart-Disease Dataset DR')
pyplot.ylabel('Mean Accuracy 10-fold CV')
fig2.savefig("fig10.png")
'''
#Time optimized learner

a = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008,
                      random_state=123)



mlpPCA= MLPClassifier(hidden_layer_sizes=(6), max_iter=750, solver='sgd', momentum=0.5, learning_rate_init=.001, random_state=123)
mlpICA = MLPClassifier(hidden_layer_sizes=(3), max_iter=750, solver='sgd', momentum=1, learning_rate_init=.0008, random_state=123)
mlpRP = MLPClassifier(hidden_layer_sizes=(6), max_iter=750, solver='sgd', momentum=0.5, learning_rate_init=.0015, random_state=123)
mlpFA = MLPClassifier(hidden_layer_sizes=(9), max_iter=750, solver='sgd', momentum=0.5, learning_rate_init=.0008, random_state=123)

def optumHD(a, b, c):
    a.fit(b, y_train_heart)
    z = a.score(c, y_test_heart)
    return z

#PCA
print('times PCA (ms):')
tot = 0
for _ in range(4):
    tic=timeit.default_timer()
    optumHD(mlpPCA, principalComponents, principalComponentstest)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/4)

#ICA
print('times ICA (ms):')
tot = 0
for _ in range(4):
    tic=timeit.default_timer()
    optumHD(mlpICA, ica_, ica_test)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/4)


#RP
print('times RP (ms):')
tot = 0
for _ in range(4):
    tic=timeit.default_timer()
    optumHD(mlpRP, rp_, rp_test)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/4)

#FA
print('times FA (ms):')
tot = 0
for _ in range(4):
    tic=timeit.default_timer()
    optumHD(mlpFA, heart_red, heart_red_test)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/4)

#Org
print('times Og (ms):')
tot = 0
for _ in range(4):
    tic=timeit.default_timer()
    optumHD(a, X_train_heart, X_test_heart)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/4)

