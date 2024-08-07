#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = keras.datasets.fashion_mnist
[train_images,train_labels],[test_images,test_labels] = data.load_data()

train_images = train_images/ 255
test_images = test_images/255
class_name = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')

plt.imshow(train_images[7])
plt.imshow(train_images[7], cmap= plt.cm.binary)
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])
model.compile(optimizer="adam", loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images,train_labels, epochs= 5)


test_loss , test_acc = model.evaluate(test_images,test_labels)
print("Test Accuracy : ", test_acc)

prediction = model.predict(test_images)
for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i] , cmap=plt.cm.binary )
    plt.xlabel('Actual: ' + class_name[(test_labels[i])])
    plt.title('Prediction : ' + class_name[np.argmax(prediction[i])])
    plt.show




# In[ ]:




