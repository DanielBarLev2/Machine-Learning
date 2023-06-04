from application import l_layer_model, predict
from sklearn.model_selection import KFold
from keras.datasets import mnist
import numpy as np

# loads MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform the data into a proper vector form and shape
x_train = x_train.reshape(60000, -1).T
x_test = x_test.reshape(10000, -1).T

# standardize the data
x_train = x_train / 255
x_test = x_test / 255

# initialize the cross-validation object
k_fold = KFold(n_splits=5, shuffle=True)

# initialize parameters
parameters = []

# iterate over the folds and trains
for train_index, val_index in k_fold.split(x_train):
    # Get the training and validation data for this fold
    x_fold_train, X_fold_test = x_train[train_index], x_test[val_index]
    y_fold_train, y_fold_test = y_train[train_index], y_test[val_index]

    # setting layers dims
    layers_dims = np.array([x_fold_train.shape[0], 20, 7, 5, 10])

    # train
    parameters = l_layer_model(x_train=x_fold_train, y_train=y_fold_train, layer_dims=layers_dims, learning_rate=0.009,
                               num_iterations=1)

# predict using the parameters
accuracy = predict(x_test=x_test, y_test=y_test, parameters=parameters)

print(accuracy)