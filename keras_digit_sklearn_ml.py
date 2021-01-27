from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier

X_train = mnist.load_data()[0][0].reshape(60000, 28 * 28)
y_train = mnist.load_data()[0][1]
print(X_train.shape)
print(y_train.shape)
X_test = mnist.load_data()[1][0].reshape(10000, 28 * 28)
y_test = mnist.load_data()[1][1]
print(X_test.shape)
print(y_test.shape)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
