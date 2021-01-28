from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam

# get data
X_train = mnist.load_data()[0][0]
y_train = mnist.load_data()[0][1]
print(X_train.shape)
print(y_train.shape)
X_test = mnist.load_data()[1][0]
y_test = mnist.load_data()[1][1]
print(X_test.shape)
print(y_test.shape)

# build model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))  # input_data需要flatten
model.add(Dense(10, activation='softmax'))  # 128(神經元)輸出 --> 多(0-9)選一
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape(60000, 28 * 28), to_categorical(y_train),
          batch_size=1000,
          epochs=10,
          verbose=2,  # 顯示過程
          validation_split=0.2)
# 很多True、False
print(np.argmax(model.predict(X_test.reshape(10000, 28 * 28)), axis=1) == y_test)
# 把很多True、False average->精準度
print(np.average(np.argmax(model.predict(X_test.reshape(10000, 28 * 28)), axis=1) == y_test))
