# # save model
# json_string = model.to_json()
# with open("model.config", "w") as text_file:
#     text_file.write(json_string)
# # save W
# model.save_weights('model.weight')
# # save model and W
# model.save('model.h5')

# 模型載入
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json

# load model
with open('model.config', 'r') as text_file:
    json_string = text_file.read()
model = model_from_json(json_string)
# load weights
model.load_weights('model.weight', by_name=False)

print('config = ', model.get_config())
print('weights = ', model.get_weights())
print('summary = ', model.summary())
print('layer = ', model.get_layer(index=1).name)
for l in model.layers:
    print(l.name)
print('params = ', model.count_params())

#######################
# model.compile(loss='categorical_crossentropy') #分類
# model.compile(loss='mse') #回歸
#########################
# output layer:
# model.add(Dense(10, activation='softmax'))  # 多分類
# model.add(Dense(2, activation='sigmoid'))  # 單分類
# model.add(Dense(1))  # regression
