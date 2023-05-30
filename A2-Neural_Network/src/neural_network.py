import numpy as np

from application import l_layer_model, predict
from keras.datasets import mnist

# loads MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform the data into a proper vector form and shape
x_train = x_train.reshape(6000, -1).T
y_train = y_train.reshape(-1, 6000)
x_test = x_test.reshape(10000, -1).T
y_test = y_test.reshape(-1, 10000)

# standardize the data
x_train = x_train / 255
x_test = x_test / 255

print('Training set shape:', x_train.shape)
print('Training labels shape:', y_train.shape)
print('Test set shape:', x_test.shape)
print('Test labels shape:', y_test.shape)

# setting layers dims
layers_dims = np.array([x_train.shape[0], 20, 7, 5, 10])

parameters = l_layer_model(x_train, y_train, layers_dims, 0, learning_rate=0.009, num_iterations=100)

print("# pass 1")

# print the accuracy
predict(x_test, parameters, y_test)