from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier

X_train = mnist.load_data()[0][0]
# X_train = X_train.reshape(60000, 28 * 28)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
y_train = mnist.load_data()[0][1]
print(X_train.shape)
print(y_train.shape)

X_test = mnist.load_data()[1][0]
# X_test = X_test.reshape(60000, 28 * 28)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
y_test = mnist.load_data()[1][1]
print(X_test.shape)
print(y_test.shape)

# 0
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 0.8783
