from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

# get data
digits = datasets.load_digits()
X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(64,)))  # input_data需要flatten
model.add(Dense(10, activation='softmax'))  # 128(神經元)輸出 --> 多(0-9)選一
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, to_categorical(y_train),
          batch_size=10,  # batch_size要改
          epochs=10,
          verbose=2,  # 顯示過程
          validation_split=0.2)

print(np.average(np.argmax(model.predict(X_test), axis=1) == y_test))
