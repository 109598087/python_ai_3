import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from sklearn.preprocessing import StandardScaler

# get data
diabetes_data = pd.read_csv('diabetes.csv')
print(diabetes_data.head())

X = diabetes_data.iloc[:, 1:-2].values
y = diabetes_data.iloc[:, -1].values
# print(X.shape)
# print(y.shape)
ss = StandardScaler()
X_ss = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_ss, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # input_data需要flatten
# model.add(Dense(256, activation='relu'))  # input_data需要flatten
model.add(Dense(2, activation='softmax'))  # 128(神經元)輸出 --> 多(0-9)選一
model.summary()

model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), to_categorical(y_train),
                    batch_size=10,
                    epochs=5,
                    verbose=2,  # 顯示過程
                    validation_split=0.2)

print(np.argmax(model.predict(X_test), axis=1))
print(y_test)
# 很多True、False
print(np.argmax(model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1])), axis=1) == y_test)
# 把很多True、False average->精準度
print(np.average(np.argmax(model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1])), axis=1) == y_test))

# draw

plt.figure(figsize=(4, 4))
plt.plot(history.history['loss'], color='red')
plt.plot(history.history['val_loss'], color='blue')
plt.plot(history.history['accuracy'], color='red')
plt.plot(history.history['val_accuracy'], color='blue')
plt.show()

# TP/FP
# FN/TN
predivtions = model.predict_classes(X_test)
print(pd.crosstab(y_test, predivtions, rownames=['real'], colnames=['預測']))

# # 產生模型圖
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

# 儲存模型
from keras.models import model_from_json

# save model
json_string = model.to_json()
with open("model.config", "w") as text_file:
    text_file.write(json_string)
# save W
model.save_weights('model.weight')
# save model and W
model.save('model.h5')

print('config = ', model.get_config())
print('weights = ', model.get_weights())
print('summary = ', model.summary())
print('layer = ', model.get_layer(index=1).name)
for l in model.layers:
    print(l.name)
print('params = ', model.count_params())

# # 模型載入
# import numpy as np
# from keras.models import Sequential
# from keras.models import model_from_json
#
# # load model
# with open('model.config', 'r') as text_file:
#     json_string = text_file.read()
# model = model_from_json(json_string)
# # load weights
# model.load_weights('model.weight', by_name=False)
#
# weihgts = model.get_weights()
# print(weihgts)
