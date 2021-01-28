# save model
json_string = model.to_json()
with open("model.confug", "w") as text_file:
    text_file.write(json_string)
# save W
model.save_weights('model.weight')
# save model and W
model.save('model.h5')

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

#######################
# model.compile(loss='categorical_crossentropy') #分類
# model.compile(loss='mse') #回歸
#########################
