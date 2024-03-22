from sklearn.random_projection import SparseRandomProjection
from scipy.sparse import csr_matrix
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error


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

purse = []
'''
for j in range(10):
    bag = []
    for i in range(2,17):
        rp = SparseRandomProjection(n_components=i, random_state=j)
        rp_ = rp.fit_transform(X_train_heart)

        inverse_data = pinv(csr_matrix.toarray(rp.components_).T)
        reconstructed_data = rp_.dot(inverse_data)
        mse = mean_squared_error(X_train_heart, reconstructed_data)
        bag.append(mse)
    purse.append(bag)
print(np.std(purse, axis=0))
print(np.mean(purse, axis=0))
'''
for j in range(10):
    bag = []
    for i in range(2,28):
        rp = SparseRandomProjection(n_components=i, random_state=j)
        rp_ = rp.fit_transform(X_train)

        inverse_data = pinv(csr_matrix.toarray(rp.components_).T)
        reconstructed_data = rp_.dot(inverse_data)
        mse = ((X_train - reconstructed_data)**2).mean(axis=None)

        bag.append(mse)
    purse.append(bag)
print(np.std(purse, axis=0))
print(np.mean(purse, axis=0))

