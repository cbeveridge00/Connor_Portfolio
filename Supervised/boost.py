import pandas
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from curve import plot_learning_curve
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn.model_selection import ShuffleSplit
import timeit
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

parameters = {'n_estimators': [25, 50, 75, 100, 150],
              'base_estimator': [DecisionTreeClassifier(criterion = 'gini', max_depth=5, min_samples_split=2, class_weight = None,random_state = 245)]}

tr = AdaBoostClassifier(random_state = 245)
gsearch = GridSearchCV(tr, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

tr2 = AdaBoostClassifier(random_state = 981)
gsearch2 = GridSearchCV(tr2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

DT_best_crite = gsearch.best_params_['n_estimators']
print("For a TTT Decision Tree model, the optimal number is "+str(DT_best_crite))

DT_best_crite = gsearch2.best_params_['n_estimators']
print("For a heart Decision Tree model, the optimal number is "+str(DT_best_crite))
fig4, axes = pyplot.subplots()
fig5, axes2 = pyplot.subplots()

title = "Learning Curve (Tic-Tac-Toe)"
title2 = "Learning Curve (Heart Disease)"
# Cross validation with 15 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=168)

cv2 = ShuffleSplit(n_splits=15, test_size=0.2, random_state=789)

estimator = AdaBoostClassifier(n_estimators=50, base_estimator=DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, class_weight=None, random_state = 245), random_state = 245)

estimator2 = AdaBoostClassifier(n_estimators=50, base_estimator=DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, class_weight=None, random_state = 981), random_state = 981)

plot_learning_curve(estimator, title, X_train, y_train, axes, ylim=(0.55, 1.01),
                    cv=cv)

plot_learning_curve(estimator2, title2, X_train_heart, y_train_heart, axes2, ylim=(0.55, 1.01),
                    cv=cv2)

fig4.savefig("fig5a.png")
fig5.savefig("fig5b.png")

#Model Complexity
parameters = {
              'n_estimators': [25, 50, 75, 100, 150],
              'base_estimator': [DecisionTreeClassifier(criterion = 'gini', max_depth=5, min_samples_split=2, class_weight = None,random_state = 245)]
             }

parameters2 = {
              'n_estimators': [25, 50, 75, 100, 150],
              'base_estimator': [DecisionTreeClassifier(criterion = 'entropy', max_depth=5, min_samples_split=2, class_weight = None,random_state = 245)]
             }

tr = AdaBoostClassifier(random_state = 245)
gsearch = GridSearchCV(tr, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_
score = model.score(X_test, y_test)
print('Optimal Model Score (TTT)')
print(score)
tr2 = AdaBoostClassifier(random_state = 981)
gsearch2 = GridSearchCV(tr2, parameters2, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)

model = gsearch2.best_estimator_
score = model.score(X_test_heart, y_test_heart)
print('Optimal Model Score (HD)')
print(score)
DT_df = pandas.DataFrame(gsearch.cv_results_)


fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_n_estimators'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(25,150)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('Max Number of Trees')
pyplot.title('Tic-Tac-Toe Endgame')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig2.savefig("graph_boost.png")

DT_df = pandas.DataFrame(gsearch2.cv_results_)


fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_n_estimators'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(25,150)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('Max Number of Trees')
pyplot.title('Heart-Disease Endgame')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig2.savefig("graph_boost_heart.png")



#Time optimized learner
def optumTTT():
    base = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, class_weight=None,
                                  random_state = 245)
    a = AdaBoostClassifier(n_estimators=75, base_estimator=base, random_state = 245)
    a.fit(X_train, y_train)
    b = a.score(X_test, y_test)
    return b


def optumHD():
    base = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, class_weight=None,
                                  random_state=981)
    a = AdaBoostClassifier(n_estimators=25, base_estimator=base, random_state = 981)
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
