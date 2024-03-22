import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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


#3D PCA on heart
pca = PCA(random_state=10)
principalComponents = pca.fit_transform(X_train_heart)
ytrain = y_train_heart.tolist()
principalComponents = np.array(principalComponents)

# Plot the ground truth - heart
fig2 = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig2, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=ytrain, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('P1')
ax2.set_ylabel('P2')
ax2.set_zlabel('P3')
ax2.set_title('Principle Components - Heart Disease')
ax2.dist = 11

fig2.savefig('pcaheart.png')
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(sum(pca.explained_variance_ratio_[:12]))
#3D PCA on game
pca = PCA(random_state=10)
principalComponents = pca.fit_transform(X_train)
ytrain = y_train.tolist()
principalComponents = np.array(principalComponents)
# Plot the ground truth - heart
fig1 = plt.figure(figsize=(12, 8))
ax2 = Axes3D(fig1, rect=[0, 0, 1, 1], elev=48, azim=134)

# Reorder the labels to have colors matching the cluster results

ax2.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=ytrain, edgecolor='k')


ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('P1')
ax2.set_ylabel('P2')
ax2.set_zlabel('P3')
ax2.set_title('Principle Components - TicTacToe')
ax2.dist = 11

fig1.savefig('pcagame.png')
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_[:15]))
