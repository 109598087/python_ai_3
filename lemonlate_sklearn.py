import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

lemonade = pd.read_csv('lemonade.csv')
print(lemonade.head())

X = lemonade.iloc[:, -5:-1].values  # values:pd->numpy
y = lemonade.iloc[:, -1].values
# print(X.shape)
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # Return the coefficient of determination  of the prediction.


# MES mean Squared Error
# print([model.predict(X_test) - y_test, 2])
# print([pow(model.predict(X_test) - y_test, 2)])
print(np.sum([pow(model.predict(X_test) - y_test, 2)]) / len(y_test))
