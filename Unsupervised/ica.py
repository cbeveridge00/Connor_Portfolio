from sklearn.decomposition import FastICA
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis
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

for i in range(2,10):
    ica = FastICA(n_components=i, max_iter=1000, random_state=10)
    ica_ = ica.fit(X_train_heart).transform(X_train_heart)
    kur = kurtosis(ica_)
    print(kur)
    print(statistics.mean(kur))

'''


ica_ = np.array(ica_)
ytrain = y_train_heart.tolist()
#3D ICA on heart

# Plot the ground truth - heart
fig2 = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig2, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(ica_[:, 0], ica_[:, 1], ica_[:, 2], c=ytrain, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('IC1')
ax2.set_ylabel('IC2')
ax2.set_zlabel('IC3')
ax2.set_title('Independent Components - Heart Disease')
ax2.dist = 11

fig2.savefig('icaheart.png')


for i in range(2,15):
    ica = FastICA(n_components=i, max_iter=1000, random_state=10)
    ica_ = ica.fit(X_train).transform(X_train)
    kur = kurtosis(ica_)
    print(kur)
    print(statistics.mean(kur))


ica = FastICA(n_components=3, random_state=10)
ica_ = ica.fit(X_train).transform(X_train)
ica_ = np.array(ica_)
ytrain = y_train.tolist()
#3D ICA on game

# Plot the ground truth - heart
fig = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(ica_[:, 0], ica_[:, 1], ica_[:, 2], c=ytrain, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('IC1')
ax2.set_ylabel('IC2')
ax2.set_zlabel('IC3')
ax2.set_title('Independent Components - TTT')
ax2.dist = 11

fig.savefig('icagame.png')
'''
