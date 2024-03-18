#Import svm model
from sklearn import svm
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from matplotlib import pyplot
import timeit
from curve import plot_learning_curve
from sklearn.model_selection import ShuffleSplit

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

parameters = {
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
             }

svm1 = svm.SVC(random_state = 245)
gsearch = GridSearchCV(svm1, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_
score = model.score(X_test, y_test)
print(score)

svm2 = svm.SVC(random_state = 981)
gsearch2 = GridSearchCV(svm2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

model = gsearch2.best_estimator_
score = model.score(X_test_heart, y_test_heart)
print(score)

DT_best_crite = gsearch.best_params_['kernel']
print("For a TTT Decision Tree model, the optimal kernel is "+str(DT_best_crite))

DT_best_balance = gsearch2.best_params_['kernel']
print(", the optimal kernel is "+str(DT_best_balance))


# Learning curve

fig4, axes = pyplot.subplots()
fig5, axes2 = pyplot.subplots()

title = "Learning Curve (Tic-Tac-Toe)"
title2 = "Learning Curve (Heart Disease)"
# Cross validation with 15 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=168)

cv2 = ShuffleSplit(n_splits=15, test_size=0.2, random_state=789)

estimator = svm.SVC(kernel='linear', random_state = 245)

estimator2 = svm.SVC(kernel='linear', random_state = 981)

plot_learning_curve(estimator, title, X_train, y_train, axes, ylim=(0.55, 1.01),
                    cv=cv)

plot_learning_curve(estimator2, title2, X_train_heart, y_train_heart, axes2, ylim=(0.55, 1.01),
                    cv=cv2)

fig4.savefig("fig6a.png")
fig5.savefig("fig6b.png")


#model Complexity analysis
parameters = {
              'C': [0.25, 0.5, .75, 1.0, 1.25, 1.75]
             }

svm1 = svm.SVC(kernel='linear', random_state = 245)
gsearch = GridSearchCV(svm1, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_
score = model.score(X_test, y_test)
print('optimal model score(TTT):')
print(score)

svm2 = svm.SVC(kernel='linear', random_state = 981)
gsearch2 = GridSearchCV(svm2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

model = gsearch2.best_estimator_
score = model.score(X_test_heart, y_test_heart)
print('optimal model score(HD):')
print(score)

DT_best_crite = gsearch.best_params_['C']
print("For a TTT Decision Tree model, the optimal c is "+str(DT_best_crite))

DT_best_balance = gsearch2.best_params_['C']
print(", the c kernel is "+str(DT_best_balance))

DT_df = pandas.DataFrame(gsearch.cv_results_)
fig3 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_C'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(0,5)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('C')
pyplot.ylabel('Mean Accuracy 15-fold CV')
pyplot.title("Tic-Tac-Toe")
fig3.savefig("graph_SVM.png")

DT_df = pandas.DataFrame(gsearch2.cv_results_)
fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_C'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(0,5)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('Maximum layers')
pyplot.ylabel('Mean Accuracy 15-fold CV')
pyplot.title("Heart Disease")
fig2.savefig("graph_SVM_heart.png")


#Time optimized learner
def optumTTT():

    a = svm.SVC(kernel='linear', C=0.25, random_state=245)
    a.fit(X_train, y_train)
    b = a.score(X_test, y_test)
    return b


def optumHD():

    a = svm.SVC(kernel='linear', C=0.75, random_state=981)
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