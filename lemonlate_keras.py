import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

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

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(Dense(1))  # for regression
model.summary()
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=10, epochs=5, verbose=0)
mse, mae = model.evaluate(X_test, y_test)
print("MSE_test: ", mse)

# draw
loss_history = history.history['loss']
plt.figure(figsize=(4, 4))
plt.plot(loss_history)
plt.show()

########
# softmax: for 多分類
# sigmoid: for 單分類
# 直接Dense(1): for regression
