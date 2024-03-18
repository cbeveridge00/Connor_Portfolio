import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn import tree
from curve import plot_learning_curve
import timeit
from matplotlib import pyplot

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

#Inital gridsearch - tictactoe
parameters = {
              'criterion': ('gini', 'entropy'),
              'min_samples_split':[2,3,4,5,6,7,8,9],
              'class_weight': ('balanced', None),
             }



tr = tree.DecisionTreeClassifier(random_state = 245)
gsearch = GridSearchCV(tr, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)

tr2 = tree.DecisionTreeClassifier(random_state = 981)
gsearch2 = GridSearchCV(tr2, parameters, cv = 15, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)


DT_best_crite = gsearch.best_params_['criterion']
print("For a TTT Decision Tree model, the optimal criterion is "+str(DT_best_crite))
DT_best_split = gsearch.best_params_['min_samples_split']
print(", the optimal min split is "+str(DT_best_split))
DT_best_balance = gsearch.best_params_['class_weight']
print(", the optimal balance is "+str(DT_best_balance))

DT_best_crite = gsearch2.best_params_['criterion']
print("For a Heart Decision Tree model, the optimal criterion is "+str(DT_best_crite))
DT_best_split = gsearch2.best_params_['min_samples_split']
print(", the optimal min split is "+str(DT_best_split))
DT_best_balance = gsearch2.best_params_['class_weight']
print(", the optimal balance is "+str(DT_best_balance))


#model Complexity Analysis - tictactoe
parameters = {
              'criterion': ['gini'],
              'min_samples_split':[2],
              'max_depth': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              'class_weight': [None],
             }

parametersh = {
              'criterion': ['entropy'],
              'min_samples_split':[2],
              'max_depth': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              'class_weight': [None],
             }

gsearch = GridSearchCV(tr, parameters, cv = 15, scoring='accuracy')
gsearch.fit(X_train, y_train)
model = gsearch.best_estimator_
score = model.score(X_test, y_test)

DT_df = pandas.DataFrame(gsearch.cv_results_)
print('Optimally Complex Model Score (TTT):')
print(score)
fig = pyplot.figure(figsize=(45,40))
_ = tree.plot_tree(model,
                   feature_names=x.columns,
                   class_names=['1','0'],
                   filled=True)
DT_best_layers = gsearch.best_params_['max_depth']
print("For a Decision Tree model, the optimal number max of layers is "+str(DT_best_layers))
fig.savefig("decistion_tree.png")


fig2 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_max_depth'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(2,20)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('Maximum layers')
pyplot.title('Tic-Tac-Toe Endgame')
pyplot.ylabel('Mean Accuracy 15-fold CV')
fig2.savefig("fig2a.png")


gsearch2 = GridSearchCV(tr2, parametersh, scoring='accuracy')
gsearch2.fit(X_train_heart, y_train_heart)
model2 = gsearch2.best_estimator_
score2 = model2.score(X_test_heart, y_test_heart)
print('Optimally Complex Model Score (HD):')
print(score2)
fig2 = pyplot.figure(figsize=(45,40))
_ = tree.plot_tree(model2,
                   feature_names=heart_x.columns,
                   class_names=['0','1'],
                   filled=True)

DT_best_layers = gsearch2.best_params_['max_depth']
print("For a heart Decision Tree model, the optimal max number of layers is "+str(DT_best_layers))
fig2.savefig("decistion_tree_heart.png")

DT_df = pandas.DataFrame(gsearch2.cv_results_)
fig3 = pyplot.figure(figsize=(12,9))
pyplot.plot(DT_df['param_max_depth'],DT_df['mean_test_score'],'g-o')
pyplot.xlim(2,20)
pyplot.ylim(0.65,1.0)
pyplot.xlabel('Maximum layers')
pyplot.ylabel('Mean Accuracy 15-fold CV')
pyplot.title("Heart Disease")
fig3.savefig("fig2b.png.png")



fig4, axes = pyplot.subplots()
fig5, axes2 = pyplot.subplots()

title = "Learning Curve (Tic-Tac-Toe)"
title2 = "Learning Curve (Heart Disease)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=168)

cv2 = ShuffleSplit(n_splits=15, test_size=0.2, random_state=789)

estimator = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_split=2, class_weight = None,random_state = 245)

estimator2 = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_split=2, class_weight = None,random_state = 981)

plot_learning_curve(estimator, title, X_train, y_train, axes, ylim=(0.55, 1.01),
                    cv=cv)

plot_learning_curve(estimator2, title2, X_train_heart, y_train_heart, axes2, ylim=(0.55, 1.01),
                    cv=cv2)

fig4.savefig("fig1a.png")
fig5.savefig("fig1b.png")



#Time optimized learner
def optumTTT():
    a = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2, class_weight=None, random_state=245, max_depth = 10)
    a.fit(X_train, y_train)
    b = a.score(X_test, y_test)
    return b


def optumHD():
    a = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2, class_weight=None, random_state=245, max_depth = 9)
    a.fit(X_train_heart, y_train_heart)
    b = a.score(X_test_heart, y_test_heart)
    return b

print('Final Test Accuracies:')
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
