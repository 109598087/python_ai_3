from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# get data
boston_price_data = datasets.load_boston()
X = boston_price_data.data
y = boston_price_data.target
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(1))
model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=15, epochs=100, verbose=1)
print(model.predict(X_test)[:10])
print(y_test[:10])
mse, mae = model.evaluate(X_test, y_test)
print("MSE_test: ", mse)

# draw
loss_history = history.history['loss']
plt.figure(figsize=(4, 4))
plt.plot(loss_history)
plt.show()
