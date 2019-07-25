# NarinoTut
Files for Tensorflow and Machine Learning Tutorial. 
##Python 
https://www.codecademy.com/learn/learn-python
```python
print("Hello World")

#variables in python
x = 5
y = "John"
print(x)
print(y)

#lists in python
thislist = ["apple", "banana", "cherry"]
print(thislist)


a = 33
b = 200

#if statement
if b > a:
  print("b is greater than a")
  
  #while loop
i = 1
while i < 6:
  print(i)
  i += 1
  
  
#functions in python  
def my_function():
  print("Hello from a function")

my_function()
```

##History of AI


##Machine Learning
least squares
supervised vs unsupervised
support Vector machines
genetic algorithms 
k-means classification

##Neural Networks
http://www.cs.us.es/~fsancho/?e=72
tensorboard info: https://www.tensorflow.org/guide/summaries_and_tensorboard
tensorboard command: tensorboard --logdir=ENTERLOGFOLDERPATH --host localhost --port 8088
#biological Intutition and History
#hyperparameters
#activation function 
#gradient descent
(stochastic vs batch)
#backpropagation
#cost functions
#one hot encoding
#regularization
(L2 regularization , early stopping, dropout, augmentation)
```python
#first model test
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
```

##Deep Neural Networks and Image Classification
Good Lectures
https://www.youtube.com/watch?v=5v1JnYv_yWs&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
Link to google colab notebooks 
Image Classification:https://www.tensorflow.org/beta/tutorials/keras/basic_classification
CNN introduction:https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
#New Methods ie GANS
#LSTM
##Image Classification 
###datasets
###Filters
###dropout 
###pooling
###Image Augementation
```python
#CNN example from Tensorflow examples
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

#additional dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

```

#Object Detection 

##Optional 
explodinng vs vanishing graidents
object detection 
custom modeling production 


#Appendix







