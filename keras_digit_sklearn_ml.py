from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
X = digits['data']
y = digits['target']

# # draw
# plt.imshow(X[0].reshape(8, 8), 'gray_r')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X, y)
# print(X_train.shape, X_test.shape)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# 取機率最大的
# print(model.predict_proba(X_test))
# print(np.argmax(model.predict_proba(X_test), axis=1)[:10])  # axis=1: 從row中選出max的位置(機率最大的)

print(np.average(model.predict(X_test) == y_test))
