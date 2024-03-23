import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

(data, target) = sklearn.datasets.load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=432)
detreeReg = DecisionTreeRegressor()
detreeReg.fit(x_train, y_train)
print(y_test)
print(detreeReg.predict(x_test))

