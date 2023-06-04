from application import l_layer_model, predict
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

def run():
    x_train, y_train, x_test, y_test = load_dataset()

    x_train, x_test = normalize(x_train=x_train, x_test=x_test)

    parameters = evaluate_model(x_data=x_train, y_data=y_train)

    accuracy = predict(x_test=x_test, y_test=y_test, parameters=parameters)

    print(f'test accuracy: {accuracy} %')


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    load mnist dataset and vectorize it and encode y data as one-hot.
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], 28*28))
    x_test = x_test.reshape((x_test.shape[0], 28*28))

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def normalize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    scale and normalise date.
    """
    x_train_norm = x_train.astype('float32')
    x_test_norm = x_test.astype('float32')

    x_train_norm = x_train_norm / 255
    x_test_norm = x_test_norm / 255

    return x_train_norm, x_test_norm


def evaluate_model(x_data: np.ndarray, y_data: np.ndarray, n_folds=5):
     """
     prepare cross validation and evaluate model.
     """

     k_fold = KFold(n_folds, shuffle=True, random_state=1)

     parameters = []

     # enumerate splits
     for x_train_i, x_test_i in k_fold.split(x_data):

         # select rows for train and test
         x_fold_train, y_fold_train, x_test, y_test = x_data[x_train_i], y_data[x_train_i],\
             x_data[x_test_i], y_data[x_test_i]

         # set up dimensions
         layers_dims = np.array([x_fold_train.shape[1], 20, 7, 5, 10])

         # train the network by folds
         parameters = l_layer_model(x_train=x_fold_train.T, y_train=y_fold_train.T, layer_dims=layers_dims,
                                    learning_rate=0.009, num_iterations=100)

         accuracy = predict(x_test=x_test.T, y_test=y_test.T, parameters=parameters)

         print(f'accuracy: {accuracy}')

     return parameters

run()