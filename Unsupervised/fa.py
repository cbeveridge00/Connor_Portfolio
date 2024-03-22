from scipy.stats import kurtosis
from sklearn.cluster import FeatureAgglomeration
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import pinv
import statistics

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
'''
fa = FeatureAgglomeration(n_clusters=3)
faa = FeatureAgglomeration(n_clusters=3, linkage='average')


game_red = fa.fit_transform(X_train)
kur = kurtosis(game_red)
print(statistics.mean(kur))
game_red = np.array(game_red)

heart_red = faa.fit_transform(X_train_heart)
kur = kurtosis(heart_red)
print(statistics.mean(kur))
heart_red = np.array(heart_red)


ytrain = y_train_heart.tolist()
ytrain2 = y_train.tolist()


fig = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(heart_red[:, 0], heart_red[:, 1], heart_red[:, 2], c=ytrain, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('FA1')
ax2.set_ylabel('FA2')
ax2.set_zlabel('FA3')
ax2.set_title('Feature Agglomeration - Heart Disease')
ax2.dist = 11


fig.savefig('faheart.png')



fig2 = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig2, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(game_red[:, 0], game_red[:, 1], game_red[:, 2], c=ytrain2, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('FA1')
ax2.set_ylabel('FA2')
ax2.set_zlabel('FA3')
ax2.set_title('Feature Agglomeration - Tic Tac Toe')
ax2.dist = 11


fig2.savefig('fagame.png')

'''
#Find best number of features
for typ in ['ward', 'average']:
    print(' ')
    for i in range(2,12):
        fa1 = FeatureAgglomeration(n_clusters=i, linkage=typ)
        fa2 = FeatureAgglomeration(n_clusters=i, linkage=typ)
        game_red = fa1.fit_transform(X_train)
        kur = kurtosis(game_red)
        print(typ + str(i))
        print(statistics.mean(kur))

        heart_red = fa2.fit_transform(X_train_heart)
        kur = kurtosis(heart_red)
        print(typ + str(i))
        print(statistics.mean(kur))