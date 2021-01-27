from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier

X_train = mnist.load_data()[0][0]
y_train = mnist.load_data()[0][1]

X_test = mnist.load_data()[1][0]
y_test = mnist.load_data()[1][1]

print(X_train.shape)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# model.score()
