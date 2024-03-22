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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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


#run clusters for each DR type
#PCA
clustererPCA = KMeans(n_clusters=4, random_state=10)
clustererPCA_EM = GaussianMixture(n_components=5, covariance_type='full', random_state=10)

PCA_kfit = clustererPCA.fit(principalComponents)
PCA_emfit = clustererPCA_EM.fit(principalComponents)

train_k = PCA_kfit.predict(principalComponents)
test_k = PCA_kfit.predict(principalComponentstest)

train_em = PCA_emfit.predict(principalComponents)
test_em = PCA_emfit.predict(principalComponentstest)


#one hot encode new values for train and test
b = np.zeros((train_k.size, train_k.max()+1))
b[np.arange(train_k.size),train_k] = 1
bk1 = StandardScaler().fit_transform(b)


b = np.zeros((test_k.size, test_k.max()+1))
b[np.arange(test_k.size),test_k] = 1
bk2 = StandardScaler().fit_transform(b)


b = np.zeros((train_em.size, train_em.max()+1))
b[np.arange(train_em.size),train_em] = 1
bem1 = StandardScaler().fit_transform(b)


b = np.zeros((test_em.size, test_em.max()+1))
b[np.arange(test_em.size),test_em] = 1
bem2 = StandardScaler().fit_transform(b)

'''
parameters = {
                'hidden_layer_sizes': [(4), (5), (6), (7), (8), (9)]
             }

a = MLPClassifier(solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(bk1, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a PCA heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(bk2, y_test_heart)
print(score2)


gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(bem1, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a Em PCA heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(bem2, y_test_heart)
print(score2)

'''
#ICA
clustererICA = KMeans(n_clusters=4, random_state=10)
clustererICA_EM = GaussianMixture(n_components=5, covariance_type='full', random_state=10)

ICA_kfit = clustererICA.fit(ica_)
ICA_emfit = clustererICA_EM.fit(ica_)

train_k = ICA_kfit.predict(ica_)
test_k = ICA_kfit.predict(ica_test)

train_em = ICA_emfit.predict(ica_)
test_em = ICA_emfit.predict(ica_test)


#one hot encode new values for train and test
b = np.zeros((train_k.size, train_k.max()+1))
b[np.arange(train_k.size),train_k] = 1
b = StandardScaler().fit_transform(b)
new_X_train_heart = np.concatenate((X_train_heart, b), axis=1)
new_X_train_heartICA = new_X_train_heart

b = np.zeros((test_k.size, test_k.max()+1))
b[np.arange(test_k.size),test_k] = 1
b = StandardScaler().fit_transform(b)
new_X_test_heart = np.concatenate((X_test_heart, b), axis=1)
new_X_test_heartICA = new_X_test_heart

b = np.zeros((train_em.size, train_em.max()+1))
b[np.arange(train_em.size),train_em] = 1
b = StandardScaler().fit_transform(b)
new_X_train_heartem = np.concatenate((X_train_heart, b), axis=1)
new_X_train_heartemICA = new_X_train_heartem

b = np.zeros((test_em.size, test_em.max()+1))
b[np.arange(test_em.size),test_em] = 1
b = StandardScaler().fit_transform(b)
new_X_test_heartem = np.concatenate((X_test_heart, b), axis=1)
new_X_test_heartemICA = new_X_test_heartem
'''
parameters = {
                'hidden_layer_sizes': [(4), (5), (6), (7), (8), (9)]
             }

a = MLPClassifier(solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(new_X_train_heart, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a ICA heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(new_X_test_heart, y_test_heart)
print(score2)


gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(new_X_train_heartem, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a Em ICA heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(new_X_test_heartem, y_test_heart)
print(score2)

'''
#RP
clustererRP = KMeans(n_clusters=3, random_state=10)
clustererRP_EM = GaussianMixture(n_components=2, covariance_type='full', random_state=10)

RP_kfit = clustererRP.fit(rp_)
RP_emfit = clustererRP_EM.fit(rp_)

train_k = RP_kfit.predict(rp_)
test_k = RP_kfit.predict(rp_test)

train_em = RP_emfit.predict(rp_)
test_em = RP_emfit.predict(rp_test)


#one hot encode new values for train and test
b = np.zeros((train_k.size, train_k.max()+1))
b[np.arange(train_k.size),train_k] = 1
b = StandardScaler().fit_transform(b)
new_X_train_heart = np.concatenate((X_train_heart, b), axis=1)
new_X_train_heartRP = new_X_train_heart

b = np.zeros((test_k.size, test_k.max()+1))
b[np.arange(test_k.size),test_k] = 1
b = StandardScaler().fit_transform(b)
new_X_test_heart = np.concatenate((X_test_heart, b), axis=1)
new_X_test_heartRP = new_X_test_heart

b = np.zeros((train_em.size, train_em.max()+1))
b[np.arange(train_em.size),train_em] = 1
b = StandardScaler().fit_transform(b)
new_X_train_heartem = np.concatenate((X_train_heart, b), axis=1)
new_X_train_heartemRP = new_X_train_heartem

b = np.zeros((test_em.size, test_em.max()+1))
b[np.arange(test_em.size),test_em] = 1
b = StandardScaler().fit_transform(b)
new_X_test_heartem = np.concatenate((X_test_heart, b), axis=1)
new_X_test_heartemRP = new_X_test_heartem
'''
parameters = {
                'hidden_layer_sizes': [(4), (5), (6), (7), (8), (9)]
             }

a = MLPClassifier(solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(new_X_train_heart, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a RP heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(new_X_test_heart, y_test_heart)
print(score2)


gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(new_X_train_heartem, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a Em RP heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(new_X_test_heartem, y_test_heart)
print(score2)
'''
#FA
clustererFA = KMeans(n_clusters=2, random_state=10)
clustererheart_redEM = GaussianMixture(n_components=2, covariance_type='full', random_state=10)

heart_redkfit = clustererFA.fit(heart_red)
heart_redemfit = clustererheart_redEM.fit(heart_red)

train_k = heart_redkfit.predict(heart_red)
test_k = heart_redkfit.predict(heart_red_test)

train_em = heart_redemfit.predict(heart_red)
test_em = heart_redemfit.predict(heart_red_test)


#one hot encode new values for train and test
b = np.zeros((train_k.size, train_k.max()+1))
b[np.arange(train_k.size),train_k] = 1
b = StandardScaler().fit_transform(b)
new_X_train_heart = np.concatenate((X_train_heart, b), axis=1)
new_X_train_heartFA = new_X_train_heart

b = np.zeros((test_k.size, test_k.max()+1))
b[np.arange(test_k.size),test_k] = 1
b = StandardScaler().fit_transform(b)
new_X_test_heart = np.concatenate((X_test_heart, b), axis=1)
new_X_test_heartFA = new_X_test_heart

b = np.zeros((train_em.size, train_em.max()+1))
b[np.arange(train_em.size),train_em] = 1
b = StandardScaler().fit_transform(b)
new_X_train_heartem = np.concatenate((X_train_heart, b), axis=1)
new_X_train_heartemFA = new_X_train_heartem

b = np.zeros((test_em.size, test_em.max()+1))
b[np.arange(test_em.size),test_em] = 1
b = StandardScaler().fit_transform(b)
new_X_test_heartem = np.concatenate((X_test_heart, b), axis=1)
new_X_test_heartemFA = new_X_test_heartem

'''
parameters = {
                'hidden_layer_sizes': [(4), (5), (6), (7), (8), (9)]
             }

a = MLPClassifier(solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(new_X_train_heart, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a FA heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(new_X_test_heart, y_test_heart)
print(score2)


gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(new_X_train_heartem, y_train_heart)

DT_best_crite = gsearch.best_params_['hidden_layer_sizes']
print("For a Em FA heart ANN model, the optimal unit is "+str(DT_best_crite))

model2 = gsearch.best_estimator_
score2 = model2.score(new_X_test_heartem, y_test_heart)
print(score2)




'''

a = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=1000, learning_rate_init=.0008,
                      random_state=123)

mlp_k_PCA = MLPClassifier(hidden_layer_sizes=(5), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)
mlp_em_PCA = MLPClassifier(hidden_layer_sizes=(4), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

mlp_k_ICA = MLPClassifier(hidden_layer_sizes=(9), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)
mlp_em_ICA = MLPClassifier(hidden_layer_sizes=(8), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)


mlp_k_RP = MLPClassifier(hidden_layer_sizes=(9), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)
mlp_em_RP = MLPClassifier(hidden_layer_sizes=(5), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

mlp_k_FA = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)
mlp_em_FA = MLPClassifier(hidden_layer_sizes=(6), solver='sgd', momentum=0.75, max_iter=750, learning_rate_init=.0008, random_state=123)

'''
parameters = {
                'max_iter': [25, 50, 75, 100, 200, 300, 400, 500, 600, 700]
             }

gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(X_train_heart, y_train_heart)
DT_Org = pandas.DataFrame(gsearch.cv_results_)

gsearchPCA = GridSearchCV(mlp_k_PCA, parameters, cv = 10, scoring='accuracy')
gsearchPCA.fit(bk1, y_train_heart)
DT_PCA = pandas.DataFrame(gsearchPCA.cv_results_)

gsearchICA = GridSearchCV(mlp_k_ICA, parameters, cv = 10, scoring='accuracy')
gsearchICA.fit(new_X_train_heartICA, y_train_heart)
DT_ICA = pandas.DataFrame(gsearchICA.cv_results_)

gsearchRP = GridSearchCV(mlp_k_RP, parameters, cv = 10, scoring='accuracy')
gsearchRP.fit(new_X_train_heartRP, y_train_heart)
DT_RP = pandas.DataFrame(gsearchRP.cv_results_)

gsearchFA = GridSearchCV(mlp_k_FA, parameters, cv = 10, scoring='accuracy')
gsearchFA.fit(new_X_train_heartFA, y_train_heart)
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
pyplot.title('Heart-Disease Dataset DR - K-means')
pyplot.ylabel('Mean Accuracy 10-fold CV')
fig2.savefig("fig11.png")

#em
gsearch = GridSearchCV(a, parameters, cv = 10, scoring='accuracy')
gsearch.fit(X_train_heart, y_train_heart)
DT_Org = pandas.DataFrame(gsearch.cv_results_)

gsearchPCA = GridSearchCV(mlp_k_PCA, parameters, cv = 10, scoring='accuracy')
gsearchPCA.fit(bem1, y_train_heart)
DT_PCA = pandas.DataFrame(gsearchPCA.cv_results_)

gsearchICA = GridSearchCV(mlp_k_ICA, parameters, cv = 10, scoring='accuracy')
gsearchICA.fit(new_X_train_heartemICA, y_train_heart)
DT_ICA = pandas.DataFrame(gsearchICA.cv_results_)

gsearchRP = GridSearchCV(mlp_k_RP, parameters, cv = 10, scoring='accuracy')
gsearchRP.fit(new_X_train_heartemRP, y_train_heart)
DT_RP = pandas.DataFrame(gsearchRP.cv_results_)

gsearchFA = GridSearchCV(mlp_k_FA, parameters, cv = 10, scoring='accuracy')
gsearchFA.fit(new_X_train_heartemFA, y_train_heart)
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
pyplot.title('Heart-Disease Dataset DR - K-means')
pyplot.ylabel('Mean Accuracy 10-fold CV')
fig2.savefig("fig12.png")
'''
def optumHD(a, b, c):
    a.fit(b, y_train_heart)
    z = a.score(c, y_test_heart)
    return z

#PCA
print('times PCA (ms):')
tot = 0
for _ in range(3):
    tic=timeit.default_timer()
    optumHD(mlp_k_PCA, bem1, bem2)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/3)

#PCA
print('times PCA (ms):')
tot = 0
for _ in range(3):
    tic=timeit.default_timer()
    optumHD(mlp_em_PCA, bem1, bem2)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/3)

#FA
print('times PCA (ms):')
tot = 0
for _ in range(3):
    tic=timeit.default_timer()
    optumHD(mlp_em_FA, new_X_train_heartemFA, new_X_test_heartemFA)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/3)

#FA - k
print('times PCA (ms):')
tot = 0
for _ in range(3):
    tic=timeit.default_timer()
    optumHD(mlp_k_FA, new_X_train_heartFA, new_X_test_heartFA)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/3)

#Org
print('times Og (ms):')
tot = 0
for _ in range(3):
    tic=timeit.default_timer()
    optumHD(a, X_train_heart, X_test_heart)
    toc=timeit.default_timer()
    tot += toc-tic

print(tot/3)