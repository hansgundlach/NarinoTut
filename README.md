# NarinoTut
Files for Tensorflow and Machine Learning Tutorial. 
## Python 
https://www.codecademy.com/learn/learn-python

https://www.w3schools.com/python/
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

#dictionaries
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)


thisdict["brand"]

#for loops
for x in "banana":
  print(x)

import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()


#for fizzbuzz
a = 3
b = 4
if a != b:
  print("something")
else:
  print(a%3)

```

Excercise #1

Write a program that prints the numbers from 1 to 100. But for multiples of three print "Fizz" instead of the number and for the multiples of five print "Buzz". For numbers which are multiples of both three and five print "FizzBuzz".



## Machine Learning

1. history of AI, features
2. least squares
3. supervised vs unsupervised
4. support Vector machines
5. genetic algorithms 
6. k-means classification
7. bayesian algorithms 
8. logistic regression

## Neural Networks
1. http://www.cs.us.es/~fsancho/?e=72
(good video and tutorial https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU)

(https://github.com/stephencwelch/Neural-Networks-Demystified)

tensorboard info: https://www.tensorflow.org/guide/summaries_and_tensorboard
tensorboard command: tensorboard --logdir=ENTERLOGFOLDERPATH --host localhost --port 8088

biological Intutition and History

2. hyperparameters

3. activation function

4. gradient descent and variations (momentum, conjugate, adam )

(stochastic vs batch)

5. backpropagation

6. cost functions

7. one hot encoding

8. regularization
(L2 regularization , early stopping, dropout, augmentation)
regression: out https://www.tensorflow.org/tutorials/keras/basic_regression
overfitting tutorial : https://www.tensorflow.org/beta/tutorials/keras/overfit_and_underfit

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


We may go over these questions:

https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning



## Redes neuronales profundas y clasificación de imágenes
si tu no puedes installar la TensorFlow
utilisar : https://www.kaggle.com/competitions
o 
Enlace a cuadernos de google colab
Image Classification:https://www.tensorflow.org/beta/tutorials/keras/basic_classification
CNN introduction:https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
1. buenas video conferencias
https://www.youtube.com/watch?v=5v1JnYv_yWs&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
2. Image Classification (GPU,TPU)
3. datasets
4. Filters
5. dropout 
6. pooling
7. Image Augementation
8. Transfer Learning 



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
## The New Revolution
1. Object Detection 
2. LSTM's
3. GANs
4. RNN
5. vanishing and exploding gradients 
6. word embeddings 
(http://projector.tensorflow.org)



## More
If we have time, I will show how to make an object detector and how to make a 
custom model. 

possible other https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb
