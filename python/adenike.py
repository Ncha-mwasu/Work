
# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import rcParams

data = pd.read_csv('DC_data.csv')
data = data.drop_duplicates(subset= ['entropy', 'deviation', 'MR'])
with open('clean_data.txt', 'a') as f:
                        f.write(str(data) +'\n')
training_set,test_set = train_test_split(data,test_size=0.2, random_state=20)
# prepare data for applying it to svm
x_train = training_set.iloc[:,0:2].values  # data
y_train = training_set.iloc[:,2].values  # target
x_test = test_set.iloc[:,0:2].values  # data
y_test = test_set.iloc[:,2].values  # target 
#print(x_train,y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Normalization(axis=-1, mean=None, variance=None, invert=False),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(2)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
            

history=model.fit(x_train, y_train, epochs=20)

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.plot(np.arange(1, 21), history.history['loss'], label='Loss')
plt.plot(np.arange(1, 21), history.history['accuracy'], label='Accuracy')
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
np.argmax(predictions[0])
