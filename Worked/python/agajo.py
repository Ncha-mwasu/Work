import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
import numpy as np

data = pd.read_csv('buhari1.csv')
data = data.drop_duplicates(subset= ['Longitude', 'Latitude', 'Elevation', 'MaxTemp', 'MinTemp', 'P', 'WS', 'RH', 'SR'])
training_set,test_set = train_test_split(data,test_size=0.2, random_state=20)

# prepare data for applying it to DNN
x_train = training_set.iloc[:,0:8].values  # data
y_train = training_set.iloc[:,8].values  # target
x_test = test_set.iloc[:,0:8].values  # data
y_test = test_set.iloc[:,8].values  # target 
tf.random.set_seed(42)
model = tf.keras.Sequential([
    #tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)]
)
model.compile( loss = tf.keras.losses.mae, #mae stands for mean absolute error
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['mae'])
model.fit(x_train, y_train, epochs=20)
preds = model.predict(x_test)
with open('predictions.txt', 'a') as f:
                        f.write(str(preds) +'\n')
print(str(preds))






