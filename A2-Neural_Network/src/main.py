from NeuralNetwork import NeuralNetwork
from keras.datasets import mnist
import numpy as np

if __name__ == '__main__':
    (x_train_data, y_train_data), (x_test, y_test) = mnist.load_data()

    model = NeuralNetwork(x_input=x_train_data, y_label=y_train_data, layer_dims=np.array([784, 10, 10, 10]),
                          learning_rate=0.009, epoch=50000, batch_size=16)

    model.evaluate()
